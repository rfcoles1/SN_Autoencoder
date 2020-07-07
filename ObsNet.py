from Network import *

class Obs_Network(Network):
    def __init__(self, obs_model='test', error='default'):        
        super().__init__()
       
        self.error = error

        self.losses_path = './records/losses_' + obs_model + '.pickle'
        self.checkpoint_path = './records/cp_' + obs_model

        encoded = tf.concat(self.encoder(self.en_input),axis=1)
              
        if self.error == 'mask':
            self.ae = tf.keras.models.Model([self.en_input, self.mask, self.err],\
                self.decoder(encoded), name='ae')
            self.ae.add_loss(loss_ae_mask(self.en_input, self.decoder(encoded), self.mask, self.err))
            self.ae.compile(optimizer = self.op_ae, loss = None)
        else:
            self.ae = tf.keras.models.Model(self.en_input, self.decoder(encoded), name='unmasked_ae')
            self.ae.compile(optimizer = self.op_ae, loss = loss_mse)
        
        self.reset()

    def train(self, train_epoch):
        it = 1

        losses = np.zeros(self.checkpoint)
        
        while it < train_epoch:
            if self.error == 'mask':
                batch = next(iter(self.obs_train_dataloader))
                inp = np.reshape(batch['x'], (self.batch_size, 7167,1)).numpy()
                
                mask = np.reshape(batch['x_msk'], (self.batch_size, 7167,1)).numpy()
                err = np.reshape(batch['x_err'], (self.batch_size, 7167,1)).numpy()
                loss = self.ae.train_on_batch((inp,mask, err))
            else:
                batch = next(iter(self.obs_train_dataloader))
                inp = np.reshape(batch['x'], (self.batch_size,7167,1)) 
                loss = self.ae.train_on_batch(inp, inp)

            losses[it%self.checkpoint] = loss

            if it % self.checkpoint == 0:
                self.curr_epoch += self.checkpoint
                print('Iterations %d' % self.curr_epoch) 
                print('Reconstruct Loss %f' % np.mean(losses))
           
                self.losses['iterations'].append(self.curr_epoch)
                self.losses['loss'].append(np.mean(losses))
 
                losses = np.zeros(self.checkpoint)
               
                self.save()

            it += 1

    def predict_ae(self, N = 500):
        inp,mask,err = self.get_batch_obs(N)
        pred = self.ae.predict_on_batch((inp,mask,err))
        return inp,mask,err, pred
