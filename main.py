# This is a sample Python script.

# from dataloader import KeyphraseDataLoader, KeyphraseDataIterator
import argparse
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import logging

import engine
import utils
from Encoder_Decoder.CopyRNN import CopyRNN
from preprocess.KP20DataSet import collate_fn
from utils import DEST_VALID_PATH

def evaluate_and_save_model(model, step):
    # TODO
    pass

def save_models_and_optim(file_path, model, optim, epoch):
    model_path = os.path.join(file_path, 'model_'+ epoch + '.pt')
    optim_path = os.path.join(file_path, 'optim_' + epoch + '.pt')
    torch.save(model.state_dict(), model_path)
    torch.save(optim.state_dict(), optim_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-exp_name",  type=str, help='')
    parser.add_argument("-train_filename", type=str, help='')
    parser.add_argument("-valid_filename",  type=str, help='')
    parser.add_argument("-test_filename",  type=str, help='')

    parser.add_argument("-dest_base_dir", type=str, help='')
    parser.add_argument("-vocab_path", type=str, help='')
    parser.add_argument("-vocab_size", type=int, default=50000, help='')
    parser.add_argument("-train_from", default='', type=str, help='')

    parser.add_argument("-token_field", default='title_and_abstract_tokens', type=str, help='')
    parser.add_argument("-keyphrase_field", default='keyphrase', type=str, help='')

    parser.add_argument("-epochs", type=int, default=20, help='')
    parser.add_argument("-batch_size", type=int, default=128, help='')
    parser.add_argument("-valid_batch_size", type=int, default=64, help='')
    parser.add_argument("-test_batch_size", type=int, default=64, help='')

    parser.add_argument("-dropout", type=float, default=0.2, help='')
    parser.add_argument("-grad_norm", type=float, default=0.0, help='')
    parser.add_argument("-max_grad", type=float, default=5.0, help='')
    parser.add_argument("-shuffle", action='store_false', help='')
    parser.add_argument("-teacher_forcing", action='store_false', help='')
    parser.add_argument("-beam_size", type=float, default=50, help='')
    parser.add_argument('-tensorboard_dir', type=str, default='', help='')
    parser.add_argument('-logfile', type=str, default='train_log.log', help='')
    parser.add_argument('-save_model_step', type=int, default=10000, help='')
    parser.add_argument('-early_stop_tolerance', type=int, default=100, help='')
    parser.add_argument('-train_parallel', action='store_true', help='')
    parser.add_argument('-schedule_lr', action='store_false', help='')
    parser.add_argument('-schedule_step', type=int, default=10000, help='')
    parser.add_argument('-schedule_gamma', type=float, default=0.1, help='')
    parser.add_argument('-processed', action='store_true', help='')
    parser.add_argument('-prefetch', action='store_false', help='')
    parser.add_argument('-lazy_loading', action='store_true', help='')
    parser.add_argument('-fix_batch_size', action='store_false', help='')
    parser.add_argument('-backend', type=str, default='tf', help='')

    # model specific parameter
    parser.add_argument("-lr", type=float, default=0.001, help='')
    parser.add_argument("-embed_size", type=int, default=200, help='')
    parser.add_argument("-max_oov_count", type=int, default=100, help='')
    parser.add_argument("-label_smoothing", type=float, default=0.0, help='use label smoothing or not')
    parser.add_argument("-max_src_len", type=int, default=1500, help='')
    parser.add_argument("-max_target_len", type=int, default=8, help='')
    parser.add_argument("-hidden_size", type=int, default=100, help='')
    parser.add_argument('-n_layers', type=int, default=1, help='')
    parser.add_argument("-attention_mode", type=str, default='general',
                        choices=['general', 'dot', 'concat'], help='')
    parser.add_argument("-bidirectional", action='store_false', help='')
    parser.add_argument("-copy_net", action='store_false', help='')
    parser.add_argument("-input_feeding", action='store_false', help='')
    parser.add_argument("-coverage", action='store_false', help='use to do the coverage loss')
    parser.add_argument("-share_embedding", action='store_false', help='decoder and encoder share the same embeeding, default true')
    parser.add_argument("-is_copy", action='store_false',
                        help='use copy or not, default true')

    ## statstics params
    parser.add_argument("-report_step", type=str,default=100, help='report the loss in every report_step')


    args = parser.parse_args()
    args.vocab_path = utils.DEST_VOCAB_PATH

    args.valid_filename = utils.DEST_VALID_PATH
    args.train_filename = utils.DEST_TRAIN_PATH
    args.test_filename = utils.DEST_TEST_PATH


    model = CopyRNN(args)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   args.schedule_step,
                                                   args.schedule_gamma)
    train_engine = engine.Kp20Engine(args, model, optimizer,scheduler)

    # train_dataset = NLPDataSet(utils.PATH)
    print(train_engine.vocab[30:40])

    train_dataset = train_engine.get_dataset('train')
    valid_dataset = train_engine.get_dataset('valid')
    test_dataset = train_engine.get_dataset('test')

    print('load dataset done')
    train_dataloader = train_engine.get_dataloader(train_dataset, batch_size=args.batch_size,shuffle=True,
                                                   collate_fn=collate_fn)
    valid_dataloader = train_engine.get_dataloader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True,
                                                   collate_fn=collate_fn)
    test_dataloader = train_engine.get_dataloader(test_dataset, batch_size=args.test_batch_size, shuffle=True,
                                                   collate_fn=collate_fn)

    print("start to train...")
    step = 0
    save_path = DEST_VALID_PATH[:DEST_VALID_PATH.rfind(r'\data')]
    save_dir = 'model_ckpts'
    # 创建保存路径
    model_save_path = os.path.join(save_path, save_dir)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    for epoch in range(args.epochs):
        for idx, batch in enumerate(tqdm(train_dataloader)):

            encoder_tokens = batch['encoder_tokens'].cuda()
            encoder_token_length = batch['encoder_token_length']
            encoder_input_mask = batch['src_input_mask'].cuda()

            src_with_oov_ids = batch['encoder_token_with_oov'].cuda()
            oov_max_lengths = batch['batch_max_oov_length']

            decode_inputs = batch['decoder_input'].cuda()

            final_distribution = model(encoder_tokens,encoder_token_length,encoder_input_mask,
                                                      src_with_oov_ids,oov_max_lengths, decode_inputs)
            target = batch["decoder_target"].cuda()
            target_length = batch["decoder_target_lengths"].cuda()  #[B]
            target_mask = batch["decoder_mask"].cuda()

            optimizer.zero_grad()

            probs = torch.gather(final_distribution, 2, target.unsqueeze(2)).view(target.size(0), target.size(1)) #[B, tgt_len]
            loss = -torch.log(probs)  #[B, tgt_len]\
            if args.label_smoothing:
                # TODO
                pass

            if args.coverage:
                coverage = model.decoder.coverage_vectors
                attns = model.decoder.attn_vectors

                coverage_tensor = torch.stack(coverage, dim=1)  # [B, tgt_len, src_len]
                attn_tensor = torch.stack(attns, dim=1)  # [B, tgt_len, src_len]
                coverage_loss = torch.sum(torch.min(coverage_tensor, attn_tensor),dim=2)  #[B, tgt_len]

                loss = loss + coverage_loss

            loss.masked_fill_(target_mask == 0, 0)  #[B, tgt_len]

            avg_loss = torch.mean(torch.sum(loss, dim=1)/target_length)
            if (idx + 1) % args.report_step == 0:
                print('at {} step, loss is now {}'.format(idx + 1, avg_loss.data))
            avg_loss.backward()

            clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            # if args.schedule_lr and step <= args.schedule_step:
            #     scheduler.step()

            step += 1
            if step % args.save_model_step == 0:
                torch.cuda.empty_cache()
                model_save_path_name = os.path.join(model_save_path, f"model_{step}_{args.lr}.pt")
                torch.save(model.state_dict(), model_save_path_name)
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        model_save_path_name = os.path.join(model_save_path, f"model_{epoch}_{args.lr}.pt")
        optim_save_path_name = os.path.join(model_save_path, f"model_{epoch}_{args.lr}.optim")

        torch.save(model.state_dict(), model_save_path_name)
        torch.save(optimizer.state_dict(), optim_save_path_name)
        torch.cuda.empty_cache()

        # save_models_and_optim(save_path, model,optimizer, epoch)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
