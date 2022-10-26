import cv2
import hydra
import os
import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from .data.dataset import Dataset
from .evaluators.metrics import PSNR, EdgeAccuracy
from .models.models import EdgeModel, InpaintingModel
from .utils.utils import Progbar

import torch.profiler

class EdgeConnect():
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.MODEL == 1:
            model_name = 'edge'
        elif cfg.MODEL == 2:
            model_name = 'inpaint'
        elif cfg.MODEL == 3:
            model_name = 'edge_inpaint'
        elif cfg.MODEL == 4:
            model_name = 'joint'

        self.debug = cfg.DEBUG
        self.model_name = model_name
        self.device = torch.device(cfg.DEVICE)

        self.edge_model = EdgeModel(cfg).to(self.device)
        self.inpaint_model = InpaintingModel(cfg).to(self.device)

        self.psnr = PSNR(255.0).to(self.device)
        self.edgeacc = EdgeAccuracy(cfg.LOSS.EDGE_THRESHOLD).to(self.device)

        train_flist = cfg.TRAIN_FLIST.split('/')[1]
        
        # test mode
        if self.cfg.MODE == 2:
            self.test_dataset = Dataset(cfg, cfg.TEST_FLIST, cfg.TEST_EDGE_FLIST, cfg.TEST_MASK_FLIST, augment=False, training=False)
        else:
            print(cfg.TRAIN_FLIST)
            self.train_dataset = Dataset(cfg, cfg.TRAIN_FLIST, cfg.TRAIN_EDGE_FLIST, cfg.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(cfg, cfg.VAL_FLIST, cfg.VAL_EDGE_FLIST, cfg.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(cfg.LOGGING.SAMPLE_SIZE)
        
        self.samples_path = os.path.join(cfg.PATH, 'samples')
        self.results_path = os.path.join(cfg.PATH, 'results')

        if cfg.RESULTS is not None:
            self.results_path = os.path.join(cfg.RESULTS)

        self.log_file = os.path.join(cfg.PATH, 'log_' + model_name + '.dat')
        self.log_images_folder = os.path.join(cfg.PATH, 'images')

    def load(self):
        if self.cfg.MODEL == 1:
            self.edge_model.load()

        elif self.cfg.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.cfg.MODEL == 1:
            self.edge_model.save()

        elif self.cfg.MODEL == 2 or self.cfg.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.cfg.MODEL
        max_iteration = int(float((self.cfg.SOLVER.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the cfg file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=7),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensor_log/model'),
                use_cuda=True
            ) as profiler:

                for items in train_loader:
                
                    self.edge_model.train()
                    self.inpaint_model.train()

                    with torch.profiler.record_function("move data to cuda"):
                        images, images_gray, edges, masks = self.cuda(*items)

                    # edge model
                    if model == 1:
                        # train
                        with torch.profiler.record_function("model"):
                            outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                        # metrics
                        with torch.profiler.record_function("metrics"):
                            precision, recall = self.edgeacc(edges * masks, outputs * masks)
                            logs.append(('precision', precision.item()))
                            logs.append(('recall', recall.item()))

                        # backward
                        with torch.profiler.record_function("backprop"):
                            self.edge_model.backward(gen_loss, dis_loss)
                            iteration = self.edge_model.iteration


                    # inpaint model
                    elif model == 2:
                        # train
                        outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                        outputs_merged = (outputs * masks) + (images * (1 - masks))

                        # metrics
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                        mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                        logs.append(('psnr', psnr.item()))
                        logs.append(('mae', mae.item()))

                        # backward
                        self.inpaint_model.backward(gen_loss, dis_loss)
                        iteration = self.inpaint_model.iteration


                    # inpaint with edge model
                    elif model == 3:
                        # train
                        if True or np.random.binomial(1, 0.5) > 0:
                            outputs = self.edge_model(images_gray, edges, masks)
                            outputs = outputs * masks + edges * (1 - masks)
                        else:
                            outputs = edges

                        outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                        outputs_merged = (outputs * masks) + (images * (1 - masks))

                        # metrics
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                        mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                        logs.append(('psnr', psnr.item()))
                        logs.append(('mae', mae.item()))

                        # backward
                        self.inpaint_model.backward(gen_loss, dis_loss)
                        iteration = self.inpaint_model.iteration


                    # joint model
                    else:
                        # train
                        e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                        e_outputs = e_outputs * masks + edges * (1 - masks)
                        i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                        outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                        # metrics
                        psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                        mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                        precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                        e_logs.append(('pre', precision.item()))
                        e_logs.append(('rec', recall.item()))
                        i_logs.append(('psnr', psnr.item()))
                        i_logs.append(('mae', mae.item()))
                        logs = e_logs + i_logs

                        # backward
                        self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                        self.edge_model.backward(e_gen_loss, e_dis_loss)
                        iteration = self.inpaint_model.iteration


                    with torch.profiler.record_function("logging"):
                        if iteration >= max_iteration:
                            keep_training = False
                            break

                        logs = [
                            ("epoch", epoch),
                            ("iter", iteration),
                        ] + logs

                        progbar.add(len(images), values=logs if self.cfg.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                        # log model at checkpoints
                        if self.cfg.LOGGING.LOG_INTERVAL and iteration % self.cfg.LOGGING.LOG_INTERVAL == 0:
                            self.log(logs)

                        # sample model at checkpoints
                        if self.cfg.LOGGING.SAMPLE_INTERVAL and iteration % self.cfg.LOGGING.SAMPLE_INTERVAL == 0:
                            print('SAMPLINGS')
                            self.sample()

                        # evaluate model at checkpoints
                        if self.cfg.LOGGING.EVAL_INTERVAL and iteration % self.cfg.LOGGING.EVAL_INTERVAL == 0:
                            print('\nstart eval...\n')
                            self.eval()

                        # save model at checkpoints
                        if self.cfg.LOGGING.SAVE_INTERVAL and iteration % self.cfg.LOGGING.SAVE_INTERVAL == 0:
                            self.save()
                    
                    profiler.step()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.cfg.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        # progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))


            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs


            logs = [("it", iteration), ] + logs
            # progbar.add(len(images), values=logs)

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.cfg.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            cv2.imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                cv2.imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                cv2.imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            print('VALIDATION SET IS EMPTY')
            return

        self.edge_model.eval()
        self.inpaint_model.eval()
        model = self.cfg.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)
         # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.cfg.LOGGING.SAMPLE_SIZE <= 6:
            image_per_row = 1
        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        #path = os.path.join(self.samples_path, self.model_name)
        path = self.log_images_folder
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.device) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()