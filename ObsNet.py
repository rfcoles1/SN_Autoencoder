from Network import *

class Obs_Network(Network):
    def __init__(self, obs_model='test', error='default'):        
        super().__init__()
       
        self.error = error

        self.losses_path = './records/losses_' + obs_model + '.pickle'
        self.checkpoint_path = './records/cp_' + obs_model

        op_ae = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        encoded = tf.concat(self.encoder(self.en_input),axis=1)
              
        if self.error == 'mask':
            self.mask = tf.keras.layers.Input(shape=(7167,1), name='mask')
            self.err = tf.keras.layers.Input(shape=(7167,1), name='err')
            self.ae = tf.keras.models.Model([self.en_input, self.mask, self.err], self.decoder(encoded))
            self.ae.add_loss(loss_ae_mask(self.en_input, self.decoder(encoded), self.mask, self.err))
            self.ae.compile(optimizer = op_ae, loss = None)
        else:
            self.ae = tf.keras.models.Model(self.en_input, self.decoder(encoded))
            self.ae.compile(optimizer = op_ae, loss = loss_ae)

        self.norm_data = np.load('./data/normalization_data.npz')
        self.x_mean = torch.Tensor(self.norm_data['x_mean'].astype(np.float32))
        self.x_std = torch.Tensor(self.norm_data['x_std'].astype(np.float32))
        
        self.obs_dataset = PayneObservedDataset('./data/aspcapStar_dr14.h5', obs_domain='APOGEE',\
            dataset='train', x_mean=self.x_mean, x_std=self.x_std, collect_x_mask=True)
        self.obs_train_dataloader = DataLoader(self.obs_dataset, batch_size=self.batch_size,\
            shuffle=True, drop_last=True)
        
        self.reset()

    def train(self, train_epoch):
        it = 1

        losses = np.zeros(self.verbose)
        
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

            losses[it%self.verbose] = loss
            if it % self.verbose == 0:
                self.curr_epoch += self.verbose
                print('Iterations %d' % self.curr_epoch) 
                print('Reconstruct Loss %f' % np.mean(losses))
           
                self.losses['iterations'].append(self.curr_epoch)
                self.losses['loss'].append(np.mean(losses))
 
                losses = np.zeros(self.verbose)
               
                if it % self.verbose*self.save_freq == 0:
                    self.save()

            it += 1

    def predict_ae(self, N = 500):
        inp,mask,err = self.get_batch_obs(N)
        pred = self.ae.predict_on_batch((inp,mask,err))
        return inp,mask,err pred
