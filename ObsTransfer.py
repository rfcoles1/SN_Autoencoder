from Network import *

class Obs_Network(Network):
    def __init__(self, synth_model='Synth', obs_model='test'):
        super().__init__()

        self.synth_path = './records/cp_' + synth_model
        self.losses_path = './records/losses_' + obs_model + '.pickle'
        self.checkpoint_path = './records/cp_' + obs_model

        self.encoder.compile(optimizer=self.op_enc, loss=loss_enc)
        self.decoder.compile(optimizer=self.op_dec, loss=loss_dec)
        
        encoded = tf.concat(self.encoder(self.en_input),axis=1)

        self.ae = tf.keras.models.Model([self.en_input,self.mask, self.err],\
            self.decoder(encoded), name='ae')
        self.ae.add_loss(loss_ae_mask(self.en_input, self.decoder(encoded), self.mask, self.err))

        self.ae.compile(optimizer = self.op_ae, loss=None)

        self.reset()

    def load_synth(self):
        self.encoder.load_weights(self.synth_path + 'enc')
        self.decoder.load_weights(self.synth_path + 'dec')
   

    def train_ae(self, train_epoch):
        it = 1

        recon_losses = np.zeros(self.verbose)
        param_losses = np.zeros(self.verbose)
 
        while it < train_epoch:
            synth_in = self.get_batch_synth(self.x_mean, self.x_std, \
                N=self.batch_size,perturbations=self.perturbations)
            synth_inp = np.reshape(synth_in[0].detach().numpy(),(self.batch_size,7167,1))
            synth_out = synth_in[1].numpy()

            batch = next(iter(self.obs_train_dataloader))
            inp_obs = np.reshape(batch['x'], (self.batch_size, 7167,1)).numpy()
            mask = np.reshape(batch['x_msk'], (self.batch_size, 7167,1)).numpy()
            err = np.reshape(batch['x_err'], (self.batch_size, 7167,1)).numpy()
            loss = self.ae.train_on_batch((inp_obs, mask, err))

            params_pred = self.encoder.predict_on_batch(synth_inp)[0]
            params_err = np.mean((params_pred - synth_out)**2)
            
            recon_losses[it%self.verbose] = loss
            param_losses[it%self.verbose] = params_err

            if it % self.verbose == 0:
                self.curr_epoch += self.verbose

                print('Iterations %d' % self.curr_epoch)
                print('Reconstruct Loss %f' % np.mean(recon_losses))
                print('Parameter Loss %f' % np.mean(param_losses))

                self.losses['iterations'].append(self.curr_epoch)
                self.losses['recon_loss'].append(np.mean(recon_losses))
                self.losses['param_loss'].append(np.mean(param_losses))
                
                recon_losses = np.zeros(self.verbose)
                param_losses = np.zeros(self.verbose)
                if it % self.verbose*self.save_freq == 0:
                    self.save()

            it += 1
     
    def train_ae_synth(self, train_epoch):
        it = 1

        recon_losses = np.zeros(self.verbose)
        param_losses = np.zeros(self.verbose)
 
        while it < train_epoch:
            synth_in = self.get_batch_synth(self.x_mean, self.x_std, \
                N=self.batch_size,perturbations=self.perturbations)
            synth_inp = np.reshape(synth_in[0].detach().numpy(),(self.batch_size,7167,1))
            synth_out = synth_in[1].numpy()

            batch = next(iter(self.obs_train_dataloader))
            inp_obs = np.reshape(batch['x'], (self.batch_size, 7167,1)).numpy()
            mask = np.reshape(batch['x_msk'], (self.batch_size, 7167,1)).numpy()
            err = np.reshape(batch['x_err'], (self.batch_size, 7167,1)).numpy()
            loss = self.ae.train_on_batch((inp_obs, mask, err))

            params_err = self.encoder.train_on_batch(synth_inp, \
                [synth_out,np.zeros([self.batch_size,self.num_z])])

            recon_losses[it%self.verbose] = loss
            param_losses[it%self.verbose] = params_err[1]

            if it % self.verbose == 0:
                self.curr_epoch += self.verbose

                print('Iterations %d' % self.curr_epoch)
                print('Reconstruct Loss %f' % np.mean(recon_losses))
                print('Parameter Loss %f' % np.mean(param_losses))

                self.losses['iterations'].append(self.curr_epoch)
                self.losses['recon_loss'].append(np.mean(recon_losses))
                self.losses['param_loss'].append(np.mean(param_losses))
                
                recon_losses = np.zeros(self.verbose)
                param_losses = np.zeros(self.verbose)
                if it % self.verbose*self.save_freq == 0:
                    self.save()

            it += 1
            
    def predict_enc(self, N = 500):
        batch = self.get_batch_synth(N = N)
        inp = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        true = batch[1].numpy()

        pred = self.encoder.predict_on_batch(inp)[0]
        return inp, true, pred

    def predict_dec(self, N = 500):
        batch = self.get_batch_synth(N = N)

        true = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        inp = batch[1].numpy()
        inp = np.concatenate([inp, np.zeros([N,2])],axis=1)

        pred = self.decoder.predict_on_batch(inp)
        return inp, true, pred 
