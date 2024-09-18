import argparse
import json
import os
import PyWave
import sched
import socket
import struct
import time
import threading

from data.header import AudioConfig, AnimConfig, TextConfig, ActionConfig, make_sgc_header
from data.package import SGCSocket

class RepeatedTimer(object):
    def __init__(self, interval, function, max_time, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.max_time = max_time
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start_time = time.time()
        self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        try:
            self.function(*self.args, **self.kwargs)
        except (socket.error, BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            is_running = False
            return

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            #print(self.next_call - self.start_time)
            if self.next_call - self.start_time > self.max_time: # set a time limit
                self.is_running = False
                return
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class live_link_test_ue():
    def __init__(self, args) -> None:
        '''
        Initialize the sockets and open the wave and blendshape files
        '''
        self.all_blendshapes_sent = False
        self.all_audio_sent = False
        self.sample_chunk_size = args.chunk_size
        self.frame_time = args.frame_time
        self.bs_fps = args.bs_fps
        self.stream_start_time = None
        self.audio_delay = args.audio_delay
        self.blendshape_delay = args.blendshape_delay
        self.max_time = args.max_time
        self.socket = None

        self.send_audio = args.send_audio
        self.send_anim = args.send_anim
        self.send_text = args.send_text
        self.send_action = args.send_action

        if not self.send_audio:
            self.all_audio_sent = True

        if not self.send_anim:
            self.all_blendshapes_sent = True

        # SET UP SOCKET PORT CONNECTION
        self.socket = SGCSocket(args.remote_address, args.port)

        # SET UP TEST FILES
        audio_fpath = args.wave_file
        a2f_json_fpath = args.json_blendshape_file
        # open and format wave data
        self.wf = PyWave.open(audio_fpath)
        # Format:
        #- WAVE_FORMAT_PCM (1)
        #- WAVE_FORMAT_IEEE_FLOAT (3)
        #- WAVE_FORMAT_ALAW
        #- WAVE_FORMAT_MULAW
        #- WAVE_FORMAT_EXTENSIBLE
        print(f"rate: {self.wf.frequency}, channels: {self.wf.channels}, bits per sample: {self.wf.bits_per_sample}, format type: {self.wf.format}")
        # open the blendshape data file
        self.a2f_json_file = open(a2f_json_fpath)
        self.a2f_json_data = json.load(self.a2f_json_file)
        print(f"BlendShapes FPS: {self.bs_fps}")

        # SET UP CONGIGS
        audio_config = AudioConfig(self.wf.frequency, self.wf.channels, self.wf.bits_per_sample, self.wf.format) if self.send_audio else None
        anim_config = AnimConfig(self.bs_fps if self.frame_time == 0 else 0.0) if self.send_anim else None
        text_config = TextConfig() if self.send_text else None
        action_config = ActionConfig() if self.send_action else None
        self.sgc_header = make_sgc_header(self.send_audio, self.send_anim, self.send_text, self.send_action, audio_config, anim_config, text_config, action_config)
        print(f"SGC Header: {self.sgc_header}")

    def send_frame_data(self):
        '''
        Callled by a repeating timer to transmit the wave and blendshape data to Unreal, frame by frame
        '''
        try:
            # | indicates a longer time than the designated frame time, . means it was at or below
            current_time = time.time()
            last_frame_time = current_time - self.last_time
            if last_frame_time < self.frame_time * 0.9:
                print(".", end="", flush=True)
            elif last_frame_time > self.frame_time * 1.1:
                print("|", end="", flush=True)
            else:
                print("-", end="", flush=True)

            self.last_time = time.time()

            # send audio data --- account for the audio_start delay
            if self.send_audio and current_time - self.stream_start_time > self.audio_delay:
                if not self.all_audio_sent:
                    self.audio_sample_data = self.wf.read_samples(self.sample_chunk_size)
                    if self.audio_sample_data:
                        self.socket.send_with_validation(self.audio_sample_data, False, b'\x01')

                # Print an A when the audio data is completely sent
                if self.audio_data_size == self.wf.tell() and not self.all_audio_sent:
                    print("A", end="", flush=True)
                    self.all_audio_sent = True
            
            # send blendshape data
            current_time = time.time()
            if self.send_anim and current_time - self.stream_start_time > self.blendshape_delay:
                if str(self.blendshape_frame_counter) in self.a2f_json_data.keys():        
                    out_data = self.a2f_json_data[str(self.blendshape_frame_counter)]
                    out_data = out_data[str("Audio2Face")] # TODO: Remove Audio2Face from JSON
                    frame_data = json.dumps(out_data, separators=(",", ":"))
                    self.socket.send_with_validation(frame_data, True, b'\x02')
                    self.blendshape_frame_counter += 1
                elif not self.all_blendshapes_sent:
                    # Print a B when the blendshape data is completely sent
                    print("B", end="", flush=True)
                    self.all_blendshapes_sent = True
        except (socket.error, BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            raise

    def initiate_data_transfer(self):
        '''
        Initiates a repeating timer that will send the data to Unreal
        '''
        try:
            self.audio_data_size = (self.wf.bits_per_sample/8) * self.wf.samples
            self.last_time = time.time()
            self.blendshape_frame_counter = 0

            self.stream_start_time = self.last_time
            self.socket.send_with_validation(self.sgc_header, True)

            if self.frame_time < 0.001:
                print("a", end="", flush=True)
                print("b", end="", flush=True)
                while not (self.all_audio_sent and self.all_blendshapes_sent):
                    self.send_frame_data()
            else:
                rt = RepeatedTimer(self.frame_time, self.send_frame_data, self.max_time)  # it auto-starts, no need of rt.start()
                try:
                    # Sleep a little longer to ensure all of the data is pushed
                    sleep_time = min((self.wf.samples / self.wf.frequency) * 1.1 + self.audio_delay + self.blendshape_delay, self.max_time*1.1)
                    print(f"Waiting for {sleep_time} seconds")

                    # if we're streaming blendshapes there's no reason to send a blendshape header
                    self.blendshapes_header_sent = True
                    print("a", end="", flush=True)
                    print("b", end="", flush=True)
                    
                    time.sleep(sleep_time)  # your long-running job goes here...
                finally:
                    rt.stop()  # better in a try/finally block to make sure the program ends!     
            self.socket.send_eos()
            print("E", end="", flush=True)
        except (socket.error, BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            return

    def close(self):
        '''
        Close all of the open files and sockets
        '''
        # close audio and blendshape files
        self.wf.close()
        self.a2f_json_file.close()

        if self.socket:
            print('\nclosing sgc socket')
            self.socket.close()


if __name__ == "__main__":
    default_audio_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "audio_violet_pcm.wav")
    #default_audio_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "1_wayne_0_1_1.wav")
    default_a2f_json_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bs_violet.json")

    parser = argparse.ArgumentParser(description="A2F LiveLink Unreal Test Stub",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Network configs
    parser.add_argument("-r", "--remote-addr", dest="remote_address", action="store", default="localhost", required=False, help="Unreal network name/IP address")
    parser.add_argument("-p", "--port", dest="port", action="store", default=12130, type=int, required=False, help="SGC socket port")

    # Testing from local files
    parser.add_argument("-w", "--wave-file", dest="wave_file", action="store", default=default_audio_fpath, required=False, help="Wave file path")
    parser.add_argument("-j", "--json-blendshape-file", dest="json_blendshape_file", action="store", default=default_a2f_json_fpath, required=False, help="Blendshape JSON file path")
    parser.add_argument("-t", "--max-time", dest="max_time", action="store", default=1e9, type=int, required=False, help="Max seconds to play audio and blendshape")
    
    # Audio Network Configs
    parser.add_argument("-au", "--send-audio", dest="send_audio", action="store_true", default=False, required=False, help="Pass this to send audio data")
    parser.add_argument("-c", "--chunk-size", dest="chunk_size", action="store", default=534, type=int, required=False, help="Audio sample chunk size sent per frame")
    parser.add_argument("-d", "--audio-delay", dest="audio_delay", action="store", default=0.0, type=float, required=False, help="Seconds delay to wait before sending audio")
    
    # Animation Network Configs
    parser.add_argument("-an", "--send-anim", dest="send_anim", action="store_true", default=False, required=False, help="Pass this to send animation data")
    parser.add_argument("-f", "--frame_time", dest="frame_time", action="store", default=0.03333333333333333, type=float, required=False, help="Time in seconds between sending blendshape frames - set to 0 to burst")
    parser.add_argument("-fps", "--blendshape-fps", dest="bs_fps", action="store", default=30.0, type=float, required=False, help="FPS value in blendshape JSON data")
    parser.add_argument("-e", "--blendshape-delay", dest="blendshape_delay", action="store", default=0.0, type=float, required=False, help="Seconds delay to wait before sending blendshapes")
    
    # Text Network Configs
    parser.add_argument("-te", "--send-text", dest="send_text", action="store_true", default=False, required=False, help="Pass this to send text data")

    # Action Network Configs
    parser.add_argument("-ac", "--send-action", dest="send_action", action="store_true", default=False, required=False, help="Pass this to send action data")

    args = parser.parse_args()

    live_link_inst = live_link_test_ue(args)
    live_link_inst.initiate_data_transfer()
    live_link_inst.close()
