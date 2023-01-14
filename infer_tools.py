import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import mel_processing

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

class vits:
    def __init__(self):
        self.hps = None
        self.net_g = None
        self.device = torch.device("cpu")

    def set_device(self, device):
        self.device = torch.device(device)
        if self.net_g != None:
            net_g.to(self.device)

    def loadCheckpoint(self, name):
        self.hps = utils.get_hparams_from_file(f"checkpoints/{name}/config.json")
        if self.hps.data.n_speakers == 0:
            self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **self.hps.model).to(self.device)
        else:
            self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model).to(self.device)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(f"checkpoints/{name}/model.pth", self.net_g, None)

    def infer(self, text, sid=None, noise_scale=1, noise_scale_w=1., length_scale=1):
        stn_tst = get_text(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            if self.hps.data.n_speakers == 0:
                tsid = None
            else:
                tsid = torch.LongTensor([self.hps.speakers.index(sid)]).to(self.device)
            audio = self.net_g.infer(x_tst, 
                x_tst_lengths, 
                sid = tsid, 
                noise_scale = noise_scale, 
                noise_scale_w = noise_scale_w, 
                length_scale = length_scale)[0][0,0].data.cpu().float().numpy()
        return (self.hps.data.sampling_rate, audio)

    def VC(self, audio, src_cha, dst_cha):
        src_speaker_id = torch.LongTensor([int(self.hps.speakers.index(src_cha))]).to(self.device)
        tgt_speaker_id = torch.LongTensor([int(self.hps.speakers.index(dst_cha))]).to(self.device)
        audio, sampling_rate = utils.load_wav_to_torch(audio)
        if sampling_rate != self.hps.data.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(sampling_rate, self.hps.data.sampling_rate))
        else:
            audio_norm = audio / 32768
            audio_norm = audio_norm.unsqueeze(0)
            spec = mel_processing.spectrogram_torch(
                audio_norm, 1024, self.hps.data.sampling_rate, 256, 1024, center=False).to(self.device)
            spec_length = torch.LongTensor([spec.data.shape[2]]).to(self.device)
            audio = net_g.voice_conversion(spec, spec_length,
                                            sid_src=src_speaker_id, sid_tgt=tgt_speaker_id)[0][0, 0].data.cpu().float().numpy()
            audio = (audio * 32768.0).squeeze().astype('int16')
            return (self.hps.data.sampling_rate, audio)

