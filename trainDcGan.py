from data import load_data
from model.dcGanModel import DCGAN 
from params.params import Params
from utils import get_sample_image
import time
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    params = Params('DCGAN')
    dataset = load_data(params)
    datasize = len(dataset)
    print('Creat Model : DC-GAN')
    model = dcGan(params)
    model.load()
    total_iters = 0
    print("Start to train the model")
    step = 0
    for epoch in range(model.epoch,params.n_epoch+1):
        start = time.time()
        epoch_iters = 0
        print("Epoch : {}".format(epoch))
        for data,label in dataset:
            total_iters += params.batch_size
            epoch_iters += params.batch_size
            model.set_input(data)
            model.D_step()
            if step % params.n_critic == 0:
                model.G_step()
            if total_iters % params.print_freq == 0:
                model.save_loss()
                model.print_loss(model.losses)
            if total_iters % params.save_latest_freq == 0:
                print('\nSaving the latest model')
                model.save_model(epoch,'latest')
            step += 1
        if epoch % params.save_epoch_freq == 0:
            model.save_model(epoch,'latest')
            model.save_model(epoch)
            img = get_sample_image(model.netG,epoch,params)
        end = time.time()
        print('\n{}/{} is Done! Time Taken : {:.4f}s'.format(epoch,params.n_epoch,end-start))
        model.print_loss(model.losses)
