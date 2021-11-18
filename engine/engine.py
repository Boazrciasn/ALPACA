import os
import wandb
from torch.utils import data
import torchvision
import tqdm
from dataset_utils.dataloaders import create_datasets
from utils.config import *
from models.loss_functions import *
from utils.device_setting import device


class EngineBasic:
    def __init__(self, model, opt):
        self.opt = opt
        print("Training Engine started with the following opt: \n{}".format(self.opt.dump()))
        self.model = model
        self.loss_fn = loss_functions[self.opt.TRAIN.LOSS_FN.NAME]().to(device)

        ############  OPTIMIZER SETUP ###########################
        if self.opt.TRAIN.OPTIMIZER.NAME == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.TRAIN.OPTIMIZER.LR)
        else:
            print("enter a valid optimizer name!")
            exit()

        ############ SCHEDULER SETUP ########################
        if self.opt.TRAIN.SCHEDULER.NAME == "step_lr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             self.opt.TRAIN.SCHEDULER.STEP_SIZE,
                                                             gamma=self.opt.TRAIN.SCHEDULER.GAMMA,
                                                             last_epoch=self.opt.TRAIN.SCHEDULER.LAST_EPOCH)
        else:
            print("enter a valid scheduler name!")
            exit()

        train_set, test_set = create_datasets(self.opt.DATA.ROOT, self.opt.DATA.SAMPLING, self.opt.DATA.INPUT, self.opt.DATA.SIZE, self.opt.DATA.NUM_INSTANCES)
        self.train_loader = data.DataLoader(train_set, batch_size=self.opt.TRAIN.BATCH_SIZE,
                                            shuffle=self.opt.TRAIN.SHUFFLE)
        self.test_loader = data.DataLoader(test_set, batch_size=self.opt.TEST.BATCH_SIZE,
                                           shuffle=self.opt.TEST.SHUFFLE)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(device)
        self.train_it = self.opt.TRAIN.START_STEP

        ####### DIRECTORY SETUP: ##########
        self.opt.WANDB.LOG_DIR = osp.join(self.opt.OUTPUT_DIR, "wandb_log", self.opt.MODEL.NAME)
        os.makedirs(self.opt.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.opt.WANDB.LOG_DIR, exist_ok=True)
        os.makedirs(osp.join(self.opt.OUTPUT_DIR, "saved_models"), exist_ok=True)

        ###### WANDB SETUP:     #############
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, dir=self.opt.WANDB.LOG_DIR,
                        config=self.opt, entity=self.opt.WANDB.ENTITY, notes=self.opt.dump())

    def train(self):
        """Runs the training"""

        self.model.train()
        self.epoch = 0
        running_loss = 0.0
        while self.train_it < self.opt.TRAIN.MAX_IT:

            epoch_loss = 0
            for x, y, q in tqdm.tqdm(self.train_loader):

                loss = self.train_step(x, y)
                running_loss += loss.item()
                epoch_loss += loss.item()

                if self.train_it != 0:
                    if self.train_it % self.opt.TRAIN.SAVE_INTERVAL == 0:
                        self.do_checkpoint(self.train_it)

                    if self.train_it % self.opt.TRAIN.TEST_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.test_loader)
                        info = "Epoch: {} \tIteration: {}\t TEST ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"test-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.TRAIN_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.train_loader)
                        info = "Epoch: {} \tIteration: {}\t TRAIN ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"train-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.LOG_INTERVAL == (self.opt.TRAIN.LOG_INTERVAL - 1):
                        info = "lossEpoch: {} \tIteration: {} \tRunning Loss: {}\t LR: {}".format(self.epoch,
                                                                                                  self.train_it,
                                                                                                  running_loss / self.opt.TRAIN.LOG_INTERVAL,
                                                                                                  self.scheduler.get_lr())
                        running_loss = 0.0
                        self.log(info)
                        self.wandb.log({"Current Batch Loss": loss.item()})

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.wandb.log({"Avg. Epoch Loss": avg_epoch_loss})
            self.log("Epoch: {} \tIteration: {} \tAvg. Epoch Loss: {}".format(self.epoch, self.train_it, avg_epoch_loss))
            self.epoch += 1

    def train_step(self, x, y):
        self.train_it += 1
        self.optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def evaluate(self, eval_set):
        """Evaluates accuracy for given set of samples."""

        print("Me evaluate undead skull!")
        self.model.eval()
        true_positive = torch.tensor(0)

        for x, y, _ in tqdm.tqdm(eval_set):
            true_positive += self.test_step(x, y)

        accuracy = torch.FloatTensor([100]) * true_positive.float() / torch.FloatTensor([len(eval_set.dataset)])
        return accuracy

    def test_step(self, x, y):
        with torch.no_grad():
            y_hat = self.model(x.to(device))
            return (torch.max(y_hat, dim=1)[1] == y.to(device)).sum().item()

    def log(self, info):
        tqdm.tqdm.write(info)

    def do_checkpoint(self, num_step):

        checkpoint = {
            'num_step': num_step,
            'mpn': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/checkpoint-{}-{}.pth".format(self.wandb.run.dir, self.opt.MODEL.NAME, num_step))


class EngineCap(EngineBasic):
    def __init__(self, model, opt):
        super(EngineCap, self).__init__(model, opt)

    def train_step(self, x, y):
        self.train_it += 1
        self.optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat_pose_mean, y_hat, y_hat_pose_sigma = self.model(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def test_step(self, x, y):
        with torch.no_grad():

            y_hat_pose_mean, y_hat, y_hat_pose_sigma = self.model(x.to(device))
            return (torch.max(y_hat, dim=1)[1] == y.to(device)).sum().item()


class EngineSRCap(EngineBasic):
    def __init__(self, model, opt):
        super(EngineSRCap, self).__init__(model, opt)
        self.loss_fn = nn.NLLLoss().to(device)

    def train_step(self, x, y):
        self.train_it += 1
        self.optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat= self.model(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def test_step(self, x, y):
        with torch.no_grad():

            y_hat = self.model(x.to(device))
            return (torch.max(y_hat, dim=1)[1] == y.to(device)).sum().item()


class EngineIDARCap(EngineBasic):
    def __init__(self, model, opt):
        super(EngineIDARCap, self).__init__(model, opt)
        self.loss_fn = nn.CrossEntropyLoss().to(device)

    def train_step(self, x, y):
        self.train_it += 1
        self.optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat= self.model(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def test_step(self, x, y):
        with torch.no_grad():

            y_hat = self.model(x.to(device))
            return (torch.max(y_hat, dim=1)[1] == y.to(device)).sum().item()


class EngineNovel():
    def __init__(self, model, opt):
        # torch.autograd.set_detect_anomaly(True)
        self.opt = opt
        print("Training Engine started with the following opt: \n{}".format(self.opt.dump()))
        self.model = model
        self.decoder_model = nn.Sequential(
            nn.Linear(opt.MODEL.FEAT_SIZE * opt.DATA.NUM_CLASS, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, opt.DATA.SIZE ** 2),
            nn.Sigmoid()
        )
        self.loss_fn = loss_functions[self.opt.TRAIN.LOSS_FN.NAME]().to(device)
        params = list(self.model.parameters())
        params.extend(list(self.decoder_model.parameters()))
        if self.opt.TRAIN.OPTIMIZER.NAME == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.opt.TRAIN.OPTIMIZER.LR)
        else:
            print("enter a valid optimizer name!")
            exit()

        ############ SCHEDULER SETUP ########################
        if self.opt.TRAIN.SCHEDULER.NAME == "step_lr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             self.opt.TRAIN.SCHEDULER.STEP_SIZE,
                                                             gamma=self.opt.TRAIN.SCHEDULER.GAMMA,
                                                             last_epoch=self.opt.TRAIN.SCHEDULER.LAST_EPOCH)
        elif self.opt.TRAIN.SCHEDULER.NAME == "reduce_on_plat":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        elif self.opt.TRAIN.SCHEDULER.NAME == "None":
            self.scheduler = None
        else:
            print("enter a valid scheduler name!")
            exit()

        train_set, test_set = create_datasets(self.opt.DATA.ROOT, self.opt.DATA.SAMPLING, self.opt.DATA.INPUT,
                                              self.opt.DATA.SIZE, self.opt.DATA.NUM_INSTANCES)
        self.train_loader = data.DataLoader(train_set, batch_size=self.opt.TRAIN.BATCH_SIZE,
                                            shuffle=self.opt.TRAIN.SHUFFLE)
        self.test_loader = data.DataLoader(test_set, batch_size=self.opt.TEST.BATCH_SIZE,
                                           shuffle=self.opt.TEST.SHUFFLE)
        if not self.opt.CPU:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
                self.decoder_model = nn.DataParallel(self.decoder_model)

        self.model.to(device)
        self.decoder_model.to(device)
        self.train_it = self.opt.TRAIN.START_STEP

        ####### DIRECTORY SETUP: ##########
        self.opt.WANDB.LOG_DIR = osp.join(self.opt.OUTPUT_DIR, "wandb_log", self.opt.MODEL.NAME)
        os.makedirs(self.opt.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.opt.WANDB.LOG_DIR, exist_ok=True)
        os.makedirs(osp.join(self.opt.OUTPUT_DIR, "saved_models"), exist_ok=True)

        ###### WANDB SETUP:     #############
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, dir=self.opt.WANDB.LOG_DIR,
                        config=self.opt, entity=self.opt.WANDB.ENTITY, notes=self.opt.dump())

        self.reconstruction_loss = nn.MSELoss()
        # self.wandb.Settings.mode = "offline"

    def train_step(self, x, y):
        self.train_it += 1
        self.optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat, y_pose, y_feat = self.model(x)
        vec_mask = torch.eye(self.opt.DATA.NUM_CLASS, device=y_feat.device).index_select(dim=0, index=y.squeeze()).unsqueeze(-1)
        reconstruct = self.decoder_model((vec_mask * y_feat).flatten(-2))
        reconstruction_loss = self.reconstruction_loss(x.flatten(-3), reconstruct)
        if self.train_it % self.opt.TRAIN.LOG_INTERVAL == (self.opt.TRAIN.LOG_INTERVAL - 1):
            tt = torch.cat(
                [reconstruct[:10, :].reshape(-1, 1, self.opt.DATA.SIZE, self.opt.DATA.SIZE),
                 x[:10, -1].unsqueeze(1)], dim=0)
            g = torchvision.utils.make_grid(tt, padding=2, nrow=10)
            images = wandb.Image(g, caption="Top: Output, Bottom: Input")
            wandb.log({"reconstructs": images})

        ce_loss = self.loss_fn(y_hat.squeeze(-1), y)
        loss = ce_loss + 1e-3 * reconstruction_loss
        loss.backward()
        params = list(self.model.parameters())
        params.extend(list(self.decoder_model.parameters()))
        torch.nn.utils.clip_grad_value_(params, .75)

        self.optimizer.step()
        if self.opt.TRAIN.SCHEDULER.NAME == "step_lr":
            self.scheduler.step()

        elif self.opt.TRAIN.SCHEDULER.NAME == "reduce_on_plat":
            self.scheduler.step(loss)

        return loss, ce_loss, reconstruction_loss

    def test_step(self, x, y):
        with torch.no_grad():
            y_hat, y_pose, y_feat = self.model(x.to(device))
            return (torch.max(y_hat.squeeze(-1), dim=1)[1] == y.to(device)).sum().item()

    def train(self):
        """Runs the training"""

        self.model.train()
        self.epoch = 0

        while self.train_it < self.opt.TRAIN.MAX_IT:

            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_recon_loss = 0
            tqdm_for = tqdm.tqdm(self.train_loader)
            for x, y, q in tqdm_for:
                loss, ce_loss, reconstruct_loss = self.train_step(x, y)

                tqdm_for.set_description("RECON LOSS: {:.4f}\tCE LOSS: {:.4f} \t LR: {}\t IT: {}".format(reconstruct_loss, ce_loss,
                                                                                                self.optimizer.param_groups[0]['lr'],
                                                                                                self.train_it))
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_recon_loss += reconstruct_loss.item()
                # self.wandb.log({"grad flow": self.wandb.Image(plot_grad_flow_v2(self.model.module.named_parameters()))})

                if self.train_it != 0:
                    if self.train_it % self.opt.TRAIN.SAVE_INTERVAL == 0:
                        self.do_checkpoint(self.train_it)

                    if self.train_it % self.opt.TRAIN.TEST_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.test_loader)
                        info = "Epoch: {} \tIteration: {}\t TEST ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"test-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.TRAIN_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.train_loader)
                        info = "Epoch: {} \tIteration: {}\t TRAIN ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"train-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.LOG_INTERVAL == (self.opt.TRAIN.LOG_INTERVAL - 1):
                        self.wandb.log({"Current Batch Loss": loss.item()})
                        self.wandb.log({"Current CE Loss": ce_loss.item()})
                        self.wandb.log({"Current Recon Loss": reconstruct_loss.item()})

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            avg_epoch_ce_loss = epoch_ce_loss / len(self.train_loader)
            avg_epoch_recon_loss = epoch_recon_loss / len(self.train_loader)
            self.wandb.log({"Avg. Epoch Loss": avg_epoch_loss})
            self.wandb.log({"Avg. Epoch CE Loss": avg_epoch_ce_loss})
            self.wandb.log({"Avg. Epoch RECON Loss": avg_epoch_recon_loss})
            self.epoch += 1

    def evaluate(self, eval_set):
        """Evaluates accuracy for given set of samples."""

        print("Me evaluate undead skull!")
        self.model.eval()
        true_positive = torch.tensor(0)

        for x, y, _ in tqdm.tqdm(eval_set):
            true_positive += self.test_step(x, y)

        accuracy = torch.FloatTensor([100]) * true_positive.float() / torch.FloatTensor([len(eval_set.dataset)])
        return accuracy

    def log(self, info):
        tqdm.tqdm.write(info)

    def do_checkpoint(self, num_step):

        checkpoint = {
            'num_step': num_step,
            'mpn': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/checkpoint-{}-{}.pth".format(self.wandb.run.dir, self.opt.MODEL.NAME, num_step))


class EngineNovelContrastive(EngineNovel):
    def __init__(self, model, opt):
        super(EngineNovelContrastive, self).__init__(model, opt)

    def test_step(self, x, y):
        with torch.no_grad():
            y_hat, y_pose, y_feat, _ = self.model(x.to(device))
            return (torch.max(y_hat.squeeze(-1), dim=1)[1] == y.to(device)).sum().item()

    def train_step(self, x, y):
        self.train_it += 1
        self.optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat, y_pose, y_feat, nce_loss = self.model(x)
        nce_loss = nce_loss.mean()  # could be problematic, for both gpus it returns loss separately
        vec_mask = torch.eye(self.opt.DATA.NUM_CLASS, device=y_feat.device).index_select(dim=0, index=y.squeeze()).unsqueeze(-1)
        reconstruct = self.decoder_model((vec_mask * y_feat).flatten(-2))
        reconstruction_loss = self.reconstruction_loss(x.flatten(-3), reconstruct)
        if self.train_it % self.opt.TRAIN.LOG_INTERVAL == (self.opt.TRAIN.LOG_INTERVAL - 1):
            tt = torch.cat(
                [reconstruct[:10, :].reshape(-1, 1, self.opt.DATA.SIZE, self.opt.DATA.SIZE),
                 x[:10, -1].unsqueeze(1)], dim=0)
            g = torchvision.utils.make_grid(tt, padding=2, nrow=10)
            images = wandb.Image(g, caption="Top: Output, Bottom: Input")
            wandb.log({"reconstructs": images})

        ce_loss = self.loss_fn(y_hat.squeeze(-1), y)
        loss = ce_loss + 1e-3 * reconstruction_loss + nce_loss
        loss.backward()
        params = list(self.model.parameters())
        params.extend(list(self.decoder_model.parameters()))
        torch.nn.utils.clip_grad_value_(params, .75)

        self.optimizer.step()
        if self.opt.TRAIN.SCHEDULER.NAME != "None":
            self.scheduler.step()
        return loss, ce_loss, reconstruction_loss, nce_loss

    def train(self):
        """Runs the training"""

        self.model.train()
        self.epoch = 0

        while self.train_it < self.opt.TRAIN.MAX_IT:

            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_recon_loss = 0
            epoch_nce_loss = 0
            tqdm_for = tqdm.tqdm(self.train_loader, ncols=250)
            for x, y, q in tqdm_for:
                loss, ce_loss, reconstruct_loss, nce_loss = self.train_step(x, y)
                tqdm_for.set_description("RECON LOSS: {:.4f}\tCE LOSS: {:.4f}\tNCE LOSS: {:.4f}\tLR: {}\tIT: {}".format(reconstruct_loss, ce_loss, nce_loss,
                                                                                                                        self.optimizer.param_groups[0]['lr'],  # self.scheduler.get_last_lr(),
                                                                                                                        self.train_it))
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_recon_loss += reconstruct_loss.item()
                epoch_nce_loss += nce_loss.item()
                # self.wandb.log({"grad flow": self.wandb.Image(plot_grad_flow_v2(self.model.module.named_parameters()))})

                if self.train_it != 0:
                    if self.train_it % self.opt.TRAIN.SAVE_INTERVAL == 0:
                        self.do_checkpoint(self.train_it)

                    if self.train_it % self.opt.TRAIN.TEST_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.test_loader)
                        info = "Epoch: {} \tIteration: {}\t TEST ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"test-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.TRAIN_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.train_loader)
                        info = "Epoch: {} \tIteration: {}\t TRAIN ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"train-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.LOG_INTERVAL == (self.opt.TRAIN.LOG_INTERVAL - 1):
                        self.wandb.log({"Current Batch Loss": loss.item()})
                        self.wandb.log({"Current CE Loss": ce_loss.item()})
                        self.wandb.log({"Current Recon Loss": reconstruct_loss.item()})
                        self.wandb.log({"Current NCE Loss": nce_loss.item()})

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            avg_epoch_ce_loss = epoch_ce_loss / len(self.train_loader)
            avg_epoch_recon_loss = epoch_recon_loss / len(self.train_loader)
            avg_epoch_nce_loss = epoch_nce_loss / len(self.train_loader)
            self.wandb.log({"Avg. Epoch Loss": avg_epoch_loss})
            self.wandb.log({"Avg. Epoch CE Loss": avg_epoch_ce_loss})
            self.wandb.log({"Avg. Epoch RECON Loss": avg_epoch_recon_loss})
            self.wandb.log({"Avg. Epoch NCE Loss": avg_epoch_nce_loss})
            self.epoch += 1


class EngineNovelNoFeat():
    def __init__(self, model, opt):
        torch.autograd.set_detect_anomaly(True)
        self.opt = opt
        print("Training Engine started with the following opt: \n{}".format(self.opt.dump()))
        self.model = model
        self.loss_fn = loss_functions[self.opt.TRAIN.LOSS_FN.NAME]().to(device)
        params = list(self.model.parameters())
        if self.opt.TRAIN.OPTIMIZER.NAME == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.opt.TRAIN.OPTIMIZER.LR)
        else:
            print("enter a valid optimizer name!")
            exit()

        ############ SCHEDULER SETUP ########################
        if self.opt.TRAIN.SCHEDULER.NAME == "step_lr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             self.opt.TRAIN.SCHEDULER.STEP_SIZE,
                                                             gamma=self.opt.TRAIN.SCHEDULER.GAMMA,
                                                             last_epoch=self.opt.TRAIN.SCHEDULER.LAST_EPOCH)
        else:
            print("enter a valid scheduler name!")
            exit()

        train_set, test_set = create_datasets(self.opt.DATA.ROOT, self.opt.DATA.SAMPLING, self.opt.DATA.INPUT,
                                              self.opt.DATA.SIZE, self.opt.DATA.NUM_INSTANCES)
        self.train_loader = data.DataLoader(train_set, batch_size=self.opt.TRAIN.BATCH_SIZE,
                                            shuffle=self.opt.TRAIN.SHUFFLE)
        self.test_loader = data.DataLoader(test_set, batch_size=self.opt.TEST.BATCH_SIZE,
                                           shuffle=self.opt.TEST.SHUFFLE)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(device)
        self.train_it = self.opt.TRAIN.START_STEP

        ####### DIRECTORY SETUP: ##########
        self.opt.WANDB.LOG_DIR = osp.join(self.opt.OUTPUT_DIR, "wandb_log", self.opt.MODEL.NAME)
        os.makedirs(self.opt.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.opt.WANDB.LOG_DIR, exist_ok=True)
        os.makedirs(osp.join(self.opt.OUTPUT_DIR, "saved_models"), exist_ok=True)

        ###### WANDB SETUP:     #############
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, dir=self.opt.WANDB.LOG_DIR,
                        config=self.opt, entity=self.opt.WANDB.ENTITY, notes=self.opt.dump())

        # self.wandb.Settings.mode = "offline"

    def train_step(self, x, y):
        self.train_it += 1
        self.optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat, y_pose = self.model(x)

        loss = self.loss_fn(y_hat.squeeze(-1), y)
        loss.backward()
        params = list(self.model.parameters())
        torch.nn.utils.clip_grad_value_(params, .75)

        self.optimizer.step()
        self.scheduler.step()
        return loss

    def test_step(self, x, y):
        with torch.no_grad():
            y_hat, y_pose = self.model(x.to(device))
            return (torch.max(y_hat.squeeze(-1), dim=1)[1] == y.to(device)).sum().item()

    def train(self):
        """Runs the training"""

        self.model.train()
        self.epoch = 0

        while self.train_it < self.opt.TRAIN.MAX_IT:

            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_recon_loss = 0
            tqdm_for = tqdm.tqdm(self.train_loader)
            for x, y, q in tqdm_for:
                loss = self.train_step(x, y)
                tqdm_for.set_description("LOSS: {:.4f} \t LR: {}\t IT: {}".format(loss,
                                                                                  self.scheduler.get_last_lr(),
                                                                                  self.train_it))
                epoch_loss += loss.item()
                # self.wandb.log({"grad flow": self.wandb.Image(plot_grad_flow_v2(self.model.module.named_parameters()))})

                if self.train_it != 0:
                    if self.train_it % self.opt.TRAIN.SAVE_INTERVAL == 0:
                        self.do_checkpoint(self.train_it)

                    if self.train_it % self.opt.TRAIN.TEST_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.test_loader)
                        info = "Epoch: {} \tIteration: {}\t TEST ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"test-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.TRAIN_TEST_INTERVAL == 0:
                        accuracy = self.evaluate(self.train_loader)
                        info = "Epoch: {} \tIteration: {}\t TRAIN ACCURACY: {}".format(self.epoch, self.train_it, accuracy)
                        self.log(info)
                        self.wandb.log({"train-set acc.": accuracy})

                    if self.train_it % self.opt.TRAIN.LOG_INTERVAL == (self.opt.TRAIN.LOG_INTERVAL - 1):
                        self.wandb.log({"Current Batch Loss": loss.item()})

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.wandb.log({"Avg. Epoch Loss": avg_epoch_loss})

            self.epoch += 1

    def evaluate(self, eval_set):
        """Evaluates accuracy for given set of samples."""

        print("Me evaluate undead skull!")
        self.model.eval()
        true_positive = torch.tensor(0)

        for x, y, _ in tqdm.tqdm(eval_set):
            true_positive += self.test_step(x, y)

        accuracy = torch.FloatTensor([100]) * true_positive.float() / torch.FloatTensor([len(eval_set.dataset)])
        return accuracy

    def log(self, info):
        tqdm.tqdm.write(info)

    def do_checkpoint(self, num_step):

        checkpoint = {
            'num_step': num_step,
            'mpn': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/checkpoint-{}-{}.pth".format(self.wandb.run.dir, self.opt.MODEL.NAME, num_step))
