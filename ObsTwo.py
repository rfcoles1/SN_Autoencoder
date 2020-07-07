from Network import *

def generateTheta(N, ndim):
    theta = [w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(N,ndim))]
    return np.asarray(theta)

def generateZ(N, ndim):
    z = 2*(np.random.uniform(size=(N,ndim))-0.5)
    return z

def loss_w2(proj_synth, proj_obs, k=4):
    w2 = (tf.nn.top_k(tf.transpose(proj_synth),k=k).values - \
            tf.nn.top_k(tf.transpose(proj_obs),k=k).values)**2
    return K.mean(w2)

def loss_l2(y_true, y_pred):
    return tf.math.reduce_sum((y_pred - y_true)**2)
 
class Obs_Network(Network):
    def __init__(self, regular='w2', obs_model='test', num_z = 10):
        super().__init__(num_z)

        self.losses_path = './records/losses_' + obs_model + '.pickle'
        self.checkpoint_path = './records/cp_' + obs_model

        self.en_input_synth = tf.keras.layers.Input(shape=(7167,1))
        self.encoder_synth = tf.keras.models.Model(self.en_input_synth, \
            self.Encoder(self.en_input_synth)[0])
        
        self.en_input_obs = tf.keras.layers.Input(shape=(7167,1))
        self.encoder_obs = tf.keras.models.Model(self.en_input_obs, \
            self.Encoder(self.en_input_obs))

        self.de_input_synth = tf.keras.layers.Input(shape=(25))
        self.decoder_synth = tf.keras.models.Model(self.de_input_synth,\
            self.Decoder(self.de_input_synth))

        self.de_input_obs = tf.keras.layers.Input(shape=(25+self.num_z))
        self.decoder_obs = tf.keras.models.Model(self.de_input_obs, \
            self.Decoder(self.de_input_obs))
        
        self.encoder_synth.compile(optimizer=self.op_enc, loss=loss_mse)
        self.encoder_obs.compile(optimizer=self.op_enc, loss=loss_mse)
        self.decoder_synth.compile(optimizer=self.op_dec, loss=loss_mse)
        self.decoder_obs.compile(optimizer=self.op_dec, loss=loss_mse)

        encoded_synth = self.encoder_synth(self.en_input_synth)
        decoded_synth = self.decoder_synth(self.de_input_synth)
        encoded_obs = tf.concat(self.encoder_obs(self.en_input_obs),axis=1)
        encoded_obsnet_synthin = self.encoder_obs(self.en_input_synth)[0]
        
        self.ae_obs = tf.keras.models.Model(inputs=[self.en_input_obs, self.mask, self.err,\
            self.en_input_synth, self.de_input_synth], \
            outputs=[self.decoder_synth(encoded_synth), self.decoder_obs(encoded_obs)])
        
        ae_loss = loss_ae_mask(self.en_input_obs, self.decoder_obs(encoded_obs), self.mask, self.err)
        synth_enc_loss = loss_mse(self.de_input_synth, encoded_synth)
        synth_dec_loss = loss_mse(self.en_input_synth, decoded_synth)
        
        loss = ae_loss + synth_enc_loss + synth_dec_loss 
        

        regular_loss = loss_l2(encoded_synth, encoded_obsnet_synthin)
        #the loss is always calculated but is only used when the following condition is true
        #this is done to remove some if statements later on, #TODO clarify 

        if regular=='l2':
            loss += regular_loss

        if regular=='w2':
            L = 50
            theta = K.variable(generateTheta(L, 25))
            z = K.variable(generateZ(L, 25))
        
            proj_synth = K.dot(encoded_synth, K.transpose(theta))
            proj_obs = K.dot(self.encoder_obs(self.en_input_obs)[0], K.transpose(theta))

            regular_loss = loss_w2(proj_synth, proj_obs, self.batch_size)
            loss += regular_loss
        
        self.ae_obs.add_loss(loss)
        self.ae_obs.add_metric(ae_loss, name='ae_obs', aggregation='mean')
        self.ae_obs.add_metric(synth_enc_loss, name='enc_synth', aggregation='mean')
        self.ae_obs.add_metric(synth_dec_loss, name='dec_synth', aggregation='mean')
        self.ae_obs.add_metric(regular_loss, name='regular', aggregation='mean')
        
        self.ae_obs.compile(optimizer = self.op_ae, loss=None)

        self.reset()

    
    def train(self, train_epoch):
        it = 1
        
        synth_param_losses = np.zeros(self.checkpoint)
        synth_recon_losses = np.zeros(self.checkpoint)
        obs_recon_losses = np.zeros(self.checkpoint)
        regular_losses = np.zeros(self.checkpoint)
        total_losses = np.zeros(self.checkpoint)

        while it < train_epoch:
            synth_in = self.get_batch_synth(N=self.batch_size)
            synth_inp = np.reshape(synth_in[0].detach().numpy(),(self.batch_size,7167,1))
            synth_out = synth_in[1].numpy()

            batch = next(iter(self.obs_train_dataloader))
            inp_obs = np.reshape(batch['x'], (self.batch_size, 7167,1)).numpy()
            mask = np.reshape(batch['x_msk'], (self.batch_size, 7167,1)).numpy()
            err = np.reshape(batch['x_err'], (self.batch_size, 7167,1)).numpy()
            
            loss = self.ae_obs.train_on_batch((inp_obs, mask, err, synth_inp, synth_out))

            recon_obs_loss = loss[1]
            enc_synth_loss = loss[2]
            dec_synth_loss = loss[3]
            regular_loss = loss[4]
            
            obs_recon_losses[it%self.checkpoint] = recon_obs_loss
            synth_recon_losses[it%self.checkpoint] = dec_synth_loss
            synth_param_losses[it%self.checkpoint] = enc_synth_loss
            regular_losses[it%self.checkpoint] = regular_loss
            total_losses[it%self.checkpoint] = loss[0]
            
            if it % self.checkpoint == 0:
                self.curr_epoch += self.checkpoint

                print('Iterations %d' % self.curr_epoch)
                print('Obs Reconstruct Loss %f' % np.mean(obs_recon_losses))
                print('Synth Reconstruct Loss %f' % np.mean(synth_recon_losses))
                print('Parameter Loss %f' % np.mean(synth_param_losses))
                print('Regularization Loss %f' % np.mean(regular_losses))
                print('Total Loss %f' % np.mean(total_losses))

                self.losses['iterations'].append(self.curr_epoch)
                self.losses['obs_recon_loss'].append(np.mean(obs_recon_losses))
                self.losses['synth_recon_loss'].append(np.mean(synth_recon_losses))
                self.losses['param_loss'].append(np.mean(synth_param_losses))
                self.losses['regular_loss'].append(np.mean(regular_losses))
                self.losses['total_loss'].append(np.mean(total_losses))

                obs_recon_losses = np.zeros(self.checkpoint)
                synth_recon_losses = np.zeros(self.checkpoint)
                param_losses = np.zeros(self.checkpoint)
                regular_losses = np.zeros(self.checkpoint)
                total_losses = np.zeros(self.checkpoint)

                self.save()

            it += 1
            
    def predict_synth_enc(self, N = 500):
        batch = self.get_batch_synth(N = N)
        inp = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        true = batch[1].numpy()

        pred = self.encoder_synth.predict_on_batch(inp)
        return inp, true, pred

    def predict_synth_dec(self, N = 500):
        batch = self.get_batch_synth(N = N)

        true = np.reshape(batch[0].detach().numpy(), (N,7167,1))
        inp = batch[1].numpy()
        #inp = np.concatenate([inp, np.zeros([N,2])],axis=1)

        pred = self.decoder_synth.predict_on_batch(inp)
        return inp, true, pred 

    def predict_obs_enc(self, N=500):
        obs_train_dataloader = DataLoader(self.obs_dataset, batch_size=N,\
            shuffle=True, drop_last=True)
        batch = next(iter(obs_train_dataloader))
        inp = np.reshape(batch['x'], (N,7167,1)).numpy()
        mask = np.reshape(batch['x_msk'], (N, 7167,1)).numpy()
        err = np.reshape(batch['x_err'], (N, 7167,1)).numpy()
        
        params, latents = self.encoder_obs.predict_on_batch((inp,mask,err))
        return [inp, mask, err], [params, latents]

    def predict_obs_ae(self, N = 500):
        obs_train_dataloader = DataLoader(self.obs_dataset, batch_size=N,\
            shuffle=True, drop_last=True)
        batch = next(iter(obs_train_dataloader))
        inp = np.reshape(batch['x'], (N,7167,1)).numpy()
        mask = np.reshape(batch['x_msk'], (N, 7167,1)).numpy()
        err = np.reshape(batch['x_err'], (N, 7167,1)).numpy()

        pred = self.ae_obs.predict_on_batch((inp,mask,err))
        return batch, pred

    def lime_avg(self,N=10):
        ref = self.encoder_obs.predict_on_batch\
            ((np.zeros((1,7167,1)),np.zeros((1,7167,1)),np.zeros((1,7167,1))))[1]

        difs = np.zeros((7167,N,self.num_z))
        
        for i in range(7167):
            inp = np.zeros((1,7167,1))
            vals = np.linspace(-1,1,N)
            for j in range(N):
                inp[:,i,:] = vals[j]
                this = self.encoder_obs.predict_on_batch\
                    ((inp,np.zeros((1,7167)),np.zeros((1,7167))))[1]
                difs[i,j] = np.subtract(this,ref)
                
        return np.mean(difs,axis=1), np.std(difs,axis=1)

    def lime_abs_avg(self,N=10):
        ref = self.encoder_obs.predict_on_batch\
            ((np.zeros((1,7167,1)),np.zeros((1,7167,1)),np.zeros((1,7167,1))))[1]

        difs = np.zeros((7167,N,self.num_z))
        
        for i in range(7167):
            inp = np.zeros((1,7167,1))
            vals = np.linspace(-1,1,N)
            for j in range(N):
                inp[:,i,:] = vals[j]
                this = self.encoder_obs.predict_on_batch\
                    ((inp,np.zeros((1,7167)),np.zeros((1,7167))))[1]
                difs[i,j] = abs(np.subtract(this,ref))
                
        return np.mean(difs,axis=1), np.std(difs,axis=1)



    def lime(self, val):
        ref = self.encoder_obs.predict_on_batch\
            ((np.zeros((1,7167,1)),np.zeros((1,7167,1)),np.zeros((1,7167,1))))[1]

        difs = np.zeros((7167,1,2))
        
        for i in range(7167):
            inp = np.zeros((1,7167,1))
            inp[:,i,:] = val
            this = self.encoder_obs.predict_on_batch\
                ((inp,np.zeros((1,7167)),np.zeros((1,7167))))[1]
            difs[i] += np.subtract(this,ref)
        
        return difs
