from Network import *

class Synth_Network(Network):
    def __init__(self, synth_model='test'):        
        super().__init__()

        self.losses_path = './records/losses_' + synth_model + '.pickle'
        self.checkpoint_path = './records/cp_' + synth_model

        self.encoder.compile(optimizer=self.op_enc, loss=loss_mse)
        self.decoder.compile(optimizer=self.op_dec, loss=loss_mse)

        self.reset()

    def train(self, train_epoch, norm=True):
        it = 1
        
        ylosses = np.zeros(self.checkpoint)
        zlosses = np.zeros(self.checkpoint)
        xlosses = np.zeros(self.checkpoint)

        while it < train_epoch+1:
            train_in = self.get_batch_synth(self.batch_size, norm_param=norm, norm_spec=False)
            inp = np.reshape(train_in[0].detach().numpy(),(self.batch_size,7167,1))
            out = train_in[1].numpy()
           
            enc_loss = self.encoder.train_on_batch(inp, [out,np.zeros([self.batch_size,self.num_z])])
            dec_loss = self.decoder.train_on_batch(\
                np.concatenate([out,np.zeros([self.batch_size,self.num_z])],axis=1), inp)
           
            ylosses[it%self.checkpoint] = enc_loss[1]
            zlosses[it%self.checkpoint] = enc_loss[2]
            xlosses[it%self.checkpoint] = dec_loss
           
            if it % self.checkpoint == 0:
                self.curr_epoch += self.checkpoint

                print('Iterations %d' % self.curr_epoch) 
                print('Encoder_Params Loss %f' % np.mean(ylosses))
                print('Encoder_Latent Loss %f' % np.mean(zlosses))
                print('Decoder_Reconstruct Loss %f' % np.mean(xlosses))
           
                self.losses['iterations'].append(self.curr_epoch)
                self.losses['enc_y'].append(np.mean(ylosses))
                self.losses['enc_z'].append(np.mean(zlosses))
                self.losses['dec_x'].append(np.mean(xlosses))
 
                ylosses = np.zeros(self.checkpoint)
                zlosses = np.zeros(self.checkpoint)
                xlosses = np.zeros(self.checkpoint)
               
                self.save()

            it += 1
   

    def predict_enc(self, N = 500, norm=True):
        batch = self.get_batch_synth(N,norm)

        inp = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        true = batch[1].numpy()

        pred = self.encoder.predict_on_batch(inp)[0]
        return inp, true, pred

    def predict_dec(self, N = 500, norm=True):
        batch = self.get_batch_synth(N,norm)

        true = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        inp = batch[1].numpy()
        inp = np.concatenate([inp, np.zeros([N,self.num_z])],axis=1)

        pred = self.decoder.predict_on_batch(inp)
        return inp, true, pred

    def predict_ae_synth(self, N = 500, norm=True):
        batch = self.get_batch_synth(N,norm)

        true = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        enc = self.encoder.predict_on_batch(true)
        enc = np.concatenate(enc, axis=1)

        pred = self.decoder.predict_on_batch(enc)
        return true, pred


