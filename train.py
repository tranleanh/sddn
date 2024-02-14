import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tqdm import tqdm
import numpy as np

from core.utils import load_hazy_clean_hint_soft_trans
from core.losses import wasserstein_loss, perceptual_and_l2_loss, l2_loss, l2_loss_hint, \
                            perceptual_and_transmission_guided_loss, \
                            soft_perceptual_and_transmission_guided_loss, soft_l2_loss_hint
from core.networks import SDDN, unet_encoder_discriminator_model, gan_model
from core.networks import img_size

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam



def save_all_weights(d, g, epoch_number, current_loss):

    BASE_DIR = 'weights/'

    save_dir_g = os.path.join(BASE_DIR, 'g')
    if not os.path.exists(save_dir_g):
        os.makedirs(save_dir_g)

    save_dir_d = os.path.join(BASE_DIR, 'd')
    if not os.path.exists(save_dir_d):
        os.makedirs(save_dir_d)

    g.save_weights(os.path.join(save_dir_g, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir_d, 'discriminator_{}.h5'.format(epoch_number)), True)


def train(n_images, batch_size, epoch_num, critic_updates, teacher_weight_path, dataset_path, loss_balance_weights, learning_rate):

    data = load_hazy_clean_hint_soft_trans(dataset_path, n_images, teacher_weight_path)
    x_train, y_train, h_train, s_train, g_train = data['A'], data['B'], data['C'], data['D'], data['E']

    delta = 0.5
    loss_bal_decay_values = []
    
    for e in range(epoch_num):
        if e <= delta*epoch_num: 
            loss_bal_decay = 1 - e/(delta*epoch_num)
        else: 
            loss_bal_decay = 0

        loss_bal_decay = np.array(loss_bal_decay)
        loss_bal_decay = np.reshape(loss_bal_decay, (1,1,1))
        loss_bal_decay_values.append(loss_bal_decay)

    print("Total data:", len(y_train))


    # Define GAN Model
    g = SDDN(ch_mul=0.25)
    g.summary()
    d = unet_encoder_discriminator_model()

    inputs = Input(shape=(img_size,img_size,4))
    guides = Input(shape=(img_size,img_size,1))
    decays = Input(shape=(1,1))

    generated_image, guided_fm, generated_image_2 = g(inputs)
    outputs = d(generated_image)

    d_on_g = Model(inputs=[inputs, guides, decays], outputs=[generated_image, guided_fm, generated_image_2, outputs])


    retrain = False

    if retrain:
        d_weight_path = 'path_to_pretrained_critic_model'
        g_weight_path = 'path_to_pretrained_generator_model'
    else:
        d_weight_path = ''
        g_weight_path = ''


    # Load pre-trained weights
    if g_weight_path != '' and d_weight_path != '':
        g.load_weights(g_weight_path)
        d.load_weights(d_weight_path)

    decay_rate = learning_rate/epoch_num*0.5

    d_opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
    d_on_g_opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)


    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False


    # Define Loss Function
    loss = [perceptual_and_transmission_guided_loss(guide=guides), \
            soft_l2_loss_hint(loss_decay=decays), \
            soft_perceptual_and_transmission_guided_loss(guide=guides, loss_decay=decays), wasserstein_loss]

    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_balance_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

    for e, epoch in enumerate(range(epoch_num)):

        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []

        loss_decay = loss_bal_decay_values[epoch]

        print(f'Epoch {str(e+1)}/{str(epoch_num)}: Loss decay = {loss_decay[0,0,0]}')


        for index in tqdm(range(int(x_train.shape[0] / batch_size))):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]

            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]
            image_hint_batch = h_train[batch_indexes]
            image_soft_batch = s_train[batch_indexes]
            guide_tran_batch = g_train[batch_indexes]

            generated_images, _, __ = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch([image_blur_batch, guide_tran_batch, loss_decay], [image_full_batch, image_hint_batch, image_soft_batch, output_true_batch])
            # print(d_on_g_loss)
            d_on_g_losses.append(d_on_g_loss)

            d.trainable = True

        print("-> DLoss:", np.mean(d_losses), "- GLoss", np.mean(d_on_g_losses))

        epoch_ = epoch+1
        if epoch_ % 5 == 0:
            save_all_weights(d, g, epoch_, int(np.mean(d_on_g_losses)))


if __name__ == '__main__':

    # Train Parameters:
    n_images = 4
    batch_size = 1       # (it's better to use batch_size = 1)
    epoch_num = 200
    critic_updates = 5
    learning_rate = 1E-4
    loss_balance_weights = [10, 5, 5, 1]
    teacher_weight_path = 'path_to_pretrained_teacher_model'
    dataset_path = 'path_to_dataset'

    # Train Network
    train(n_images, batch_size, epoch_num, critic_updates, teacher_weight_path, dataset_path, loss_balance_weights, learning_rate)

