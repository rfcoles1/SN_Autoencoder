from Network import *

class Obs_Network(Network):
    def __init__(self, obs_model='test'):
        super().__init__()

        self.losses_path = './records/losses_' + obs_model + '.pickle'
        self.checkpoint_path = './records/cp_' + obs_model

        self.encoder.compile(optimizer=self.op_enc, loss=loss_enc)
        self.decoder.compile(optimizer=self.op_dec, loss=loss_dec)
        
        encoded = tf.concat(self.encoder(self.en_input),axis=1)

        self.ae = tf.keras.models.Model([self.en_input,self.mask, self.err], \
            self.decoder(encoded), name = 'ae')
        self.ae.add_loss(loss_ae_mask(self.en_input, self.decoder(encoded), self.mask, self.err))

        self.ae.compile(optimizer = self.op_ae, loss=None)

        self.reset()
    
    def train(self, train_epoch):
        it = 1

        synth_recon_losses = np.zeros(self.checkpoint)
        obs_recon_losses = np.zeros(self.checkpoint)
        param_losses = np.zeros(self.checkpoint)
 
        while it < train_epoch:
            synth_in = self.get_batch_synth(self.x_mean, self.x_std, \
                N=self.batch_size,perturbations=self.perturbations)
            synth_inp = np.reshape(synth_in[0].detach().numpy(),(self.batch_size,7167,1))
            synth_out = synth_in[1].numpy()

            batch = next(iter(self.obs_train_dataloader))
            inp_obs = np.reshape(batch['x'], (self.batch_size, 7167,1)).numpy()
            mask = np.reshape(batch['x_msk'], (self.batch_size, 7167,1)).numpy()
            err = np.reshape(batch['x_err'], (self.batch_size, 7167,1)).numpy()
            
            obs_loss = self.ae.train_on_batch((inp_obs, mask, err))

            synth_enc_loss = self.encoder.train_on_batch(synth_inp, \
                [synth_out,np.zeros([self.batch_size,self.num_z])])
            synth_dec_loss = self.decoder.train_on_batch( \
                np.concatenate([synth_out,np.zeros([self.batch_size,self.num_z])],axis=1),synth_inp)

            obs_recon_losses[it%self.checkpoint] = obs_loss
            synth_recon_losses[it%self.checkpoint] = synth_dec_loss
            param_losses[it%self.checkpoint] = synth_enc_loss[1]

            if it % self.checkpoint == 0:
                self.curr_epoch += self.checkpoint

                print('Iterations %d' % self.curr_epoch)
                print('Obs Reconstruct Loss %f' % np.mean(obs_recon_losses))
                print('Synth Reconstruct Loss %f' % np.mean(synth_recon_losses))
                print('Parameter Loss %f' % np.mean(param_losses))

                self.losses['iterations'].append(self.curr_epoch)
                self.losses['obs_recon_loss'].append(np.mean(obs_recon_losses))
                self.losses['synth_recon_loss'].append(np.mean(synth_recon_losses))
                self.losses['param_loss'].append(np.mean(param_losses))
                
                obs_recon_losses = np.zeros(self.checkpoint)
                synth_recon_losses = np.zeros(self.checkpoint)
                param_losses = np.zeros(self.checkpoint)
                
                self.save()

            it += 1
            
    def predict_enc(self, N = 500):
        batch = self.get_batch_synth(N = N)
        inp = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        true = batch[1].numpy()

        pred = self.encoder.predict_on_batch(inp)[0]
        return inp, true, pred

    def predict_synth_dec(self, N = 500):
        batch = self.get_batch_synth(N = N)

        true = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        inp = batch[1].numpy()
        inp = np.concatenate([inp, np.zeros([N,2])],axis=1)

        pred = self.decoder.predict_on_batch(inp)
        return inp, true, pred 

    def predict_synth_ae(self, N = 500):
        obs_train_dataloader = DataLoader(self.obs_dataset, batch_size=N,\
            shuffle=True, drop_last=True)
        batch = next(iter(obs_train_dataloader))
        inp = np.reshape(batch['x'], (N,7167,1)).numpy()
        mask = np.reshape(batch['x_msk'], (N, 7167,1)).numpy()
        err = np.reshape(batch['x_err'], (N, 7167,1)).numpy()

        pred = self.ae.predict_on_batch((inp,mask,err))
        return batch, pred
