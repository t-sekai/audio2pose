from dataclasses import dataclass

@dataclass
class AudioConfig:
    frequency : int
    channels : int
    bits_per_sample : int
    format : int
    def __post_init__(self):
        assert isinstance(self.frequency, int)
        assert isinstance(self.channels, int)
        assert isinstance(self.bits_per_sample, int)
        assert isinstance(self.format, int)
    def __str__(self) -> str:
        return f"|{self.frequency}:{self.channels}:{self.bits_per_sample}:{self.format}"

@dataclass
class AnimConfig:
    fps : float
    def __post_init__(self):
         assert isinstance(self.fps, float)
    def __str__(self) -> str:
        return f"|{self.fps}"

@dataclass
class TextConfig:
    def __str__(self) -> str:
        pass

@dataclass
class ActionConfig:
    def __str__(self) -> str:
        pass

def make_sgc_header(
            send_audio: bool = False,
            send_anim: bool = False,
            send_text: bool = False,
            send_action: bool = False,
            audio_config: AudioConfig = None,
            anim_config: AnimConfig = None,
            text_config: TextConfig = None,
            action_config: ActionConfig = None,
    ) -> str:
        """
        Parameters
        ----------
        send_audio - Send audio to Unreal
        send_anim - Send animation to Unreal
        send_text - Send text to Unreal
        send_action - Send action to Unreal
        audio_config - Required audio configs if send_audio
        anim_config - Required animation configs if send_anim
        text_config - Required text configs if send_text
        action_config - Required action configs if send_action
        """
        assert send_audio or send_anim or send_text or send_action, "At least one mode needs to be selected."
        if send_audio:
             assert audio_config is not None, "audio_config cannot be None when send_audio is True."
        if send_anim:
             assert anim_config is not None, "audio_config cannot be None when send_anim is True."
        if send_text:
             assert text_config is not None, "text_config cannot be None when send_text is True."
        if send_action:
             assert action_config is not None, "action_config cannot be None when send_action is True."
        
        ### SET UP BIT MASK###
        audio_bit = 1 << 0 if send_audio else 0
        anim_bit = 1 << 1 if send_anim else 0
        text_bit = 1 << 2 if send_text else 0
        action_bit = 1 << 3 if send_action else 0
        bitmask = audio_bit | anim_bit | text_bit | action_bit

        ### SET UP CONIGS STRING ###
        audio_config_str = str(audio_config) if send_audio else ""
        anim_config_str = str(anim_config) if send_anim else ""
        text_config_str = str(text_config) if send_text else ""
        action_config_str = str(action_config) if send_action else ""
        configs_str = audio_config_str + anim_config_str + text_config_str + action_config_str

        ### SET UP HEADER ###
        header = f"SGC|{bitmask}{configs_str}"
        return header