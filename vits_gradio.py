import infer_tools as ift
import gradio as gr
import os

class VitsGradio:
    def __init__(self):
        self.vits_ins = ift.vits()
        self.lspk = []
        self.modelPaths = []
        for root,dirs,files in os.walk("checkpoints"):
            for dir in dirs:
                self.modelPaths.append(dir)
        with gr.Blocks() as self.Vits:
            with gr.Tab("TextToSpeech"):
                with gr.Row() as self.TextToSpeech:
                    with gr.Column():
                        with gr.Row():
                            self.text = gr.Textbox(label = "需要转换的文本")
                            self.sid = gr.Dropdown(label = "角色", choices = self.lspk)
                        with gr.Row():
                            self.ns = gr.Slider(label = "噪声规模", maximum = 1, minimum = 0.0001, step = 0.0001, value = 0.667)
                            self.nsw = gr.Slider(label = "dp噪声规模", maximum = 1, minimum = 0.0001, step = 0.0001, value = 0.8)
                            self.ls = gr.Slider(label = "长度规模", maximum = 10, minimum = 0.0001, step = 0.0001, value = 1.0)
                        with gr.Row():
                            self.btnTTS = gr.Button("文本转语音")
                        with gr.Row():
                            self.TTSOutputs = gr.Audio()
                self.btnTTS.click(self.vits_ins.infer, inputs=[self.text,self.sid,self.ns,self.nsw,self.ls], outputs=[self.TTSOutputs])
            with gr.Tab("VoiceConversion"):
                with gr.Row(visible=False) as self.VoiceConversion:
                    with gr.Column():
                        with gr.Row():
                            self.srcaudio = gr.Audio(type = "filepath", label = "输入音频")
                        with gr.Row():
                            self.ssid = gr.Dropdown(label = "源角色", choices = self.lspk)
                            self.dsid = gr.Dropdown(label = "目标角色", choices = self.lspk)
                        with gr.Row():
                            self.btnVC = gr.Button("说话人转换")
                        with gr.Row():
                            self.VCOutputs = gr.Audio()
                self.btnVC.click(self.vits_ins.VC, inputs=[self.srcaudio,self.ssid,self.dsid], outputs=[self.VCOutputs])
            with gr.Tab("SelectModel"):
                with gr.Column():
                    modelstrs = gr.Dropdown(label = "模型", choices = self.modelPaths, value = self.modelPaths[0], type = "value")
                    devicestrs = gr.Dropdown(label = "设备", choices = ["cpu","cuda"], value = "cpu", type = "value")
                    btnMod = gr.Button("载入模型")
                    btnMod.click(self.loadModel, inputs=[modelstrs,devicestrs], outputs = [self.sid,self.ssid,self.dsid,self.VoiceConversion])

    def loadModel(self, path, device):
        self.vits_ins.set_device(device)
        self.vits_ins.loadCheckpoint(path)
        for spk in self.vits_ins.hps.speakers:
            self.lspk.append(spk)
        if self.vits_ins.hps.data.n_speakers == 0:
            self.lspk.append("SingleModel")
            VChange = gr.update(visible = False)
        else:
            VChange = gr.update(visible = True)
        SChange = gr.update(choices = self.lspk, value = self.lspk[0])
        SSChange = gr.update(choices = self.lspk, value = self.lspk[0])
        SDChange = gr.update(choices = self.lspk, value = self.lspk[0])
        return [SChange,SSChange,SDChange,VChange]

grVits = VitsGradio()

grVits.Vits.launch()