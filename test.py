import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.metric.utils import calc_cer, calc_wer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []
    argmax_cer_preds = []
    argmax_wer_preds = []
    beam_search_lm_cers = []
    beam_search_lm_wers = []

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            batch["log_probs"] = batch["log_probs"].cpu().numpy()
            argmax_cer = []
            argmax_wer = []
            beam_lm_cers = []
            beam_lm_wers = []
            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[: int(batch["log_probs_length"][i])]
                beam_search_lm = text_encoder.ctc_beam_search_from_liba(batch["log_probs"][i], batch["log_probs_length"][i], beam_size=100)[0][0]
                count_cer_argm = calc_cer(text_encoder.normalize_text(batch["text"][i]),
                                          text_encoder.ctc_decode(argmax.cpu().numpy()))
                count_wer_argm = calc_wer(text_encoder.normalize_text(batch["text"][i]),
                                          text_encoder.ctc_decode(argmax.cpu().numpy()))
                count_cer_bs_lm = calc_cer(text_encoder.normalize_text(batch["text"][i]), beam_search_lm)
                count_wer_bs_lm = calc_wer(text_encoder.normalize_text(batch["text"][i]), beam_search_lm)
                results.append(
                    {
                        "ground_trurh": text_encoder.normalize_text(batch["text"][i]),
                        "pred_text_argmax": text_encoder.ctc_decode(argmax.cpu().numpy()),
                        "pred_text_beam_search_lm": beam_search_lm,
                        "argmax cers": count_cer_argm,
                        "argmax_wers": count_wer_argm,
                        "beam_search_lm_cers": count_cer_bs_lm,
                        "beam_search_lm_wers": count_wer_bs_lm
                    }
                )
                argmax_cer.append(count_cer_argm)
                argmax_wer.append(count_wer_argm)
                beam_lm_cers.append(count_cer_bs_lm)
                beam_lm_wers.append(count_wer_bs_lm)
            argmax_cer_preds.append(sum(argmax_cer) / len(argmax_cer))
            argmax_wer_preds.append(sum(argmax_wer) / len(argmax_wer))
            beam_search_lm_cers.append(sum(beam_lm_cers) / len(beam_lm_cers))
            beam_search_lm_wers.append(sum(beam_lm_wers) / len(beam_lm_wers))

    list_of_preds = []
    list_of_preds.append({'argmax_cers_et_all': sum(argmax_cer_preds) / len(argmax_cer_preds) * 100})
    list_of_preds.append({'argmax_wers_et_all': sum(argmax_wer_preds) / len(argmax_wer_preds) * 100})
    list_of_preds.append({'beam_lm_cers_et_all': sum(beam_search_lm_cers) / len(beam_search_lm_cers) * 100})
    list_of_preds.append({'beam_lm_wers_et_all': sum(beam_search_lm_wers) / len(beam_search_lm_wers) * 100})
    list_of_preds.append(results)

    with Path(out_file).open("w") as f:
        json.dump(list_of_preds, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=5,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = ROOT_PATH / "default_test_config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
