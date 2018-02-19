from ops import *
from utils import *
from glob import glob
import time

class CycleGAN(object):
    def __init__(self, sess, args):
        self.model_name = 'CycleGAN'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset

        self.print_freq = 100
        self.decay_step = 100
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.norm = args.norm

        self.do_resnet = args.do_resnet
        self.learning_rate = args.learning_rate
        self.lambdaA = args.lambdaA
        self.lambdaB = args.lambdaB
        self.identity = args.identity
        self.beta1 = args.beta1
        self.dis_layer = args.dis_layer
        self.res_block = args.res_block

        self.pool_size = args.pool_size
        self.height = 256
        self.width = 256
        self.channel = 3

        self.trainA, self.trainB = prepare_data(dataset_name=self.dataset_name)
        self.num_batches = max(len(self.trainA) // self.batch_size, len(self.trainB) // self.batch_size)
        # may be i will use deque


    def resnet_block(self, x_init, dim, do_resnet, is_training=True, reuse=False, scope="resnet"):
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.pad(x_init, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = conv_layer(x, filter_size=dim, kernel=[3, 3], stride=1, norm=self.norm, is_training=is_training, layer_name='conv1')
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            x = conv_layer(x, filter_size=dim, kernel=[3, 3], stride=1, norm=self.norm, is_training=is_training, do_relu=False, layer_name='conv2')

            if do_resnet :
                return relu(x + x_init)
            else :
                return relu(x)

    def generator(self, x, n_blocks, is_training=True, reuse=False, scope="generator"):
        with tf.variable_scope(scope, reuse=reuse):

            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            x = conv_layer(x, filter_size=32, kernel=[7, 7], stride=1, norm=self.norm, is_training=is_training, layer_name='conv1')

            n_downsampling = 2

            for i in range(n_downsampling):
                mult = pow(2, i)
                x = conv_layer(x, filter_size=32 * mult * 2, kernel=[3, 3], stride=2, padding=1,
                               norm=self.norm, is_training=is_training, layer_name='conv' + str(i + 2))

            mult = pow(2, n_downsampling)
            for i in range(n_blocks):
                x = self.resnet_block(x, dim=32 * mult, do_resnet=self.do_resnet, is_training=is_training, reuse=reuse, scope='resblock' + str(i))

            for i in range(n_downsampling):
                mult = pow(2, (n_downsampling - i))
                x = deconv_layer(x, filter_size=(32 * mult) / 2, kernel=[3, 3], stride=2, padding=1,
                                 norm=self.norm, is_training=is_training, layer_name='deconv' + str(i))

            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            x = conv_layer(x, filter_size=3, kernel=[7, 7], norm=self.norm, is_training=is_training, do_relu=False, layer_name='conv4')
            x = tanh(x)

            return x

    def discriminator(self, x, n_layers, is_training=True, reuse=False, scope="discriminator"):
        with tf.variable_scope(scope, reuse=reuse):
            x = conv_layer(x, filter_size=64, kernel=[4, 4], stride=2, padding=1, do_norm=False, is_training=is_training, leak=0.2,
                           layer_name='conv1')

            for n in range(1, n_layers):  # n_layer=3, 1 2
                nf_mult = min(2 ** n, 8)
                x = conv_layer(x, filter_size=64 * nf_mult, kernel=[4, 4], stride=2, padding=1, norm=self.norm, is_training=is_training, leak=0.2,
                               layer_name='conv' + str(n + 1))

            nf_mult = min(2 ** n_layers, 8)
            x = conv_layer(x, filter_size=64 * nf_mult, kernel=[4, 4], stride=1, padding=1, norm=self.norm, is_training=is_training, leak=0.2,
                           layer_name='conv' + str(n_layers + 1))

            x = conv_layer(x, filter_size=1, kernel=[4, 4], stride=1, padding=1, do_norm=False, is_training=is_training, do_relu=False,
                           layer_name='conv' + str(n_layers + 2))

            return x
    def build_model(self):
        self.domain_A = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.channel], name='domain_A') # real A
        self.domain_B = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.channel], name='domain_B') # real B

        # self.fake_A_sample = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.channel], name='fake_A_sample')
        # self.fake_B_sample = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.channel], name='fake_B_sample')

        self.fake_A_pool = ImagePool(self.pool_size) # A로 generate된 이미지들의 pool
        self.fake_B_pool = ImagePool(self.pool_size) # B로 generate된 이미지들의 pool

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Define Generator, Discriminator """
        # Generator
        self.fake_B = self.generator(self.domain_A, self.res_block, is_training=True, scope='generator_B') # B'
        self.fake_A = self.generator(self.domain_B, self.res_block, is_training=True, scope='generator_A') # A'

        self.test_B = self.generator(self.domain_A, self.res_block, is_training=False, scope='generator_B')
        self.test_A = self.generator(self.domain_B, self.res_block, is_training=False, scope='generator_A')

        self.recon_A = self.generator(self.fake_B, self.res_block, is_training=True, reuse=True, scope='generator_A') # A -> B' -> A
        self.recon_B = self.generator(self.fake_A, self.res_block, is_training=True, reuse=True, scope='generator_B') # B -> A -> B

        self.identity_A = self.generator(self.domainA, self.res_block, is_training=True, reuse=True, scope='generator_A')
        self.identity_B = self.generator(self.domain_B, self.res_block, is_training=True, reuse=True, scope='generator_B')

        # Discriminator
        self.dis_real_A = self.discriminator(self.domain_A, self.dis_layer, is_training=True, scope='discriminator_A')
        self.dis_real_B = self.discriminator(self.domain_B, self.dis_layer, is_training=True, scope='discriminator_B')

        self.dis_fake_A = self.discriminator(self.fake_A, self.dis_layer, is_training=True, reuse=True, scope='discriminator_A')
        self.dis_fake_B = self.discriminator(self.fake_B, self.dis_layer, is_training=True, reuse=True, scope='discriminator_B')

        self.dis_fake_pool_A = self.discriminator(self.fake_A_pool.query(self.fake_A), self.dis_layer, is_training=True, reuse=True, scope='discriminator_A')
        self.dis_fake_pool_B = self.discriminator(self.fake_B_pool.query(self.fake_B), self.dis_layer, is_training=True, reuse=True, scope='discriminator_B')

        """ Loss Function """
        self.G_A_loss = tf.reduce_mean(tf.squared_difference(self.dis_fake_A, 1)) + \
                        self.lambdaA*(tf.reduce_mean(tf.abs(self.domain_A - self.recon_A))) + \
                        self.identity*(tf.reduce_mean(tf.abs(self.domain_A - self.identity_A)))
        self.G_B_loss = tf.reduce_mean(tf.squared_difference(self.dis_fake_B, 1)) + \
                        self.lambdaB*(tf.reduce_mean(tf.abs(self.domain_B - self.recon_B))) + \
                        self.identity*(tf.reduce_mean(tf.abs(self.domain_B - self.identity_B)))

        self.Generator_loss = self.G_A_loss + self.G_B_loss

        self.D_A_loss = (tf.reduce_mean(tf.squared_difference(self.dis_real_A, 1)) + tf.reduce_mean(tf.square(self.dis_fake_pool_A))) / 2.0
        self.D_B_loss = (tf.reduce_mean(tf.squared_difference(self.dis_real_B, 1)) + tf.reduce_mean(tf.square(self.dis_fake_pool_B))) / 2.0

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        d_A_vars = [var for var in t_vars if 'discriminator_A' in var.name]
        d_B_vars = [var for var in t_vars if 'discriminator_B' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            Adam = tf.train.AdamOptimizer(self.lr, beta1=self.beta1)

            self.G_optim = Adam.minimize(self.Generator_loss, var_list=G_vars)
            self.d_A_optim = Adam.minimize(self.D_A_loss, var_list=d_A_vars)
            self.d_B_optim = Adam.minimize(self.D_B_loss, var_list=d_B_vars)


        """" Summary """

        self.g_loss = tf.summary.scalar("G_loss", self.Generator_loss)

        self.d_A_loss_sum = tf.summary.scalar("d_A_loss", self.D_A_loss)
        self.d_B_loss_sum = tf.summary.scalar("d_B_loss", self.D_B_loss)
        self.d_loss = tf.summary.merge([self.d_A_loss_sum, self.d_B_loss_sum])

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            lr = self.learning_rate if epoch < self.decay_step else self.learning_rate * (self.epoch - epoch) / (self.epoch - self.decay_step)
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                random_index_A = np.random.choice(len(self.trainA), size=self.batch_size, replace=False)
                random_index_B = np.random.choice(len(self.trainB), size=self.batch_size, replace=False)
                batch_A_images = self.trainA[random_index_A]
                batch_B_images = self.trainB[random_index_B]

                # Update D
                _, _, summary_str = self.sess.run(
                    [self.d_A_optim, self.d_B_optim, self.d_loss],
                    feed_dict = {self.domain_A : batch_A_images, self.domain_B : batch_B_images, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                # Update G
                fake_A, fake_B, _, summary_str = self.sess.run(
                        [self.fake_A, self.fake_B, self.Generator_loss, self.g_loss],
                    feed_dict = {self.domain_A : batch_A_images, self.domain_B : batch_B_images, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time))

                if np.mod(counter, 100) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))
                    save_images(batch_B_images, [self.batch_size, 1],
                                './{}/real_B_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))

                    save_images(fake_A, [self.batch_size, 1],
                                './{}/fake_A_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))

                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                self.save(self.checkpoint_dir, counter)

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))
        
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_B, feed_dict = {self.domain_A :sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file), self.width, self.height))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path), self.width, self.height))
            index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_A, feed_dict = {self.domain_B : sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file), self.width, self.height))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path), self.width, self.height))
            index.write("</tr>")
        index.close()
