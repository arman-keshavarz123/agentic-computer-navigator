# Nova: MacOS Voice Automation Agent

Nova is a sophisticated Python-based automation agent designed for MacOS. It combines speech recognition with **Anthropic's Claude 3.5 Sonnet (Computer Use API)** to "see" your screen and control your mouse and keyboard to execute complex tasks via voice commands.

## Key Features

* **Wake Word Activation**: Listens continuously for the keyword **"Nova"** to trigger valid commands, ignoring background noise.
* **Full Desktop Control**: Utilizing `pyautogui` and AppleScript, Nova can click, drag, scroll, type, and manage application windows (using Spotlight and shortcuts).
* **Visual Context**: The agent takes real-time screenshots of your desktop to understand the state of the UI and verify that tasks are completed correctly.
* **Interruptible Workflow**: Includes a threading system that allows you to interrupt the agent mid-task by saying "Nova", "Stop", or "Wait".
* **Local Speech Processing**: Uses `faster_whisper` running locally on the CPU to transcribe audio, ensuring audio data is processed quickly without external speech-to-text APIs.

## Usage

### Starting the Agent
Once the script is running, the system initializes the Whisper model and begins listening on the default microphone.

### Voice Commands
To give a command, simply speak clearly:

> *"Nova, open Safari and find the latest news on AI."*
>
> *"Nova, open Terminal, list the files in the Documents folder, and tell me what you see."*
>
> *"Nova, move that window to the left side of the screen."*

### Interruption
If the agent is in the middle of a long task or misunderstands a command, you can cut it off immediately by saying "NOVA!" 

**To stop the agent:**
* Say **"Nova stop"**
* Say **"Wait"**
* Say **"Nova [new command]"** (This stops the current action and queues the new one).


## How It Works

1.  **Audio Loop**: A background thread captures raw audio via `sounddevice`.
2.  **VAD & Transcription**: When speech is detected above a volume threshold, it is buffered and transcribed locally using `faster_whisper`.
3.  **Planner**: If the wake word is detected, the text is sent to the Anthropic API along with a screenshot of the current desktop.
4.  **Execution**: Claude analyzes the screenshot and text, returning a specific tool use (e.g., `mouse_move`, `type`, `key`).
5.  **Action**: The script maps these tool requests to local OS events using `pyautogui`.
