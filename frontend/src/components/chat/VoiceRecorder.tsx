import React, { useState, useRef, useCallback } from 'react';
import {
  IconButton,
  Tooltip,
  Box,
  Typography,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import {
  Mic,
  MicOff,
  Stop,
  Send,
  VolumeUp,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface VoiceRecorderProps {
  onTranscript: (transcript: string) => void;
  disabled?: boolean;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onTranscript,
  disabled = false,
}) => {
  const theme = useTheme();
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recognitionRef = useRef<any>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);

  const startRecording = useCallback(async () => {
    try {
      // Check if Web Speech API is supported
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      
      if (!SpeechRecognition) {
        // Fallback to MediaRecorder API
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Set up audio visualization
        audioContextRef.current = new AudioContext();
        const source = audioContextRef.current.createMediaStreamSource(stream);
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        source.connect(analyserRef.current);
        dataArrayRef.current = new Uint8Array(analyserRef.current.frequencyBinCount);

        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          // Here you would typically send the audio to a speech-to-text service
          setIsProcessing(true);
          
          // Mock processing delay
          setTimeout(() => {
            setTranscript("I'm sorry, speech recognition is not available in this browser. Please type your message.");
            setDialogOpen(true);
            setIsProcessing(false);
          }, 1000);

          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
      } else {
        // Use Web Speech API
        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event: any) => {
          let finalTranscript = '';
          let interimTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }

          setTranscript(finalTranscript + interimTranscript);
        };

        recognition.onend = () => {
          setIsRecording(false);
          if (transcript.trim()) {
            setDialogOpen(true);
          }
        };

        recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
          setIsRecording(false);
          setTranscript("Speech recognition error. Please try again or type your message.");
          setDialogOpen(true);
        };

        recognition.start();
      }

      setIsRecording(true);
      setRecordingTime(0);
      setTranscript('');

      // Start recording timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
        
        // Update audio level visualization
        if (analyserRef.current && dataArrayRef.current) {
          analyserRef.current.getByteFrequencyData(dataArrayRef.current);
          const average = dataArrayRef.current.reduce((a, b) => a + b) / dataArrayRef.current.length;
          setAudioLevel(average / 255);
        }
      }, 100);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      setTranscript("Microphone access denied. Please enable microphone permissions and try again.");
      setDialogOpen(true);
    }
  }, [transcript]);

  const stopRecording = useCallback(() => {
    setIsRecording(false);
    
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
    }

    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }

    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  }, []);

  const handleSendTranscript = () => {
    if (transcript.trim()) {
      onTranscript(transcript.trim());
      setTranscript('');
    }
    setDialogOpen(false);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <>
      <Tooltip title={isRecording ? "Stop recording" : "Start voice recording"}>
        <span>
          <IconButton
            onClick={isRecording ? stopRecording : startRecording}
            disabled={disabled || isProcessing}
            sx={{
              color: isRecording ? theme.palette.error.main : theme.palette.action.active,
              '&:hover': {
                bgcolor: isRecording 
                  ? `${theme.palette.error.main}10` 
                  : `${theme.palette.primary.main}10`,
              },
              position: 'relative',
            }}
          >
            {isProcessing ? (
              <CircularProgress size={20} />
            ) : isRecording ? (
              <Stop />
            ) : (
              <Mic />
            )}
            
            {/* Audio level indicator */}
            {isRecording && (
              <Box
                sx={{
                  position: 'absolute',
                  inset: -2,
                  borderRadius: '50%',
                  border: 2,
                  borderColor: theme.palette.error.main,
                  opacity: 0.3 + (audioLevel * 0.7),
                  animation: 'pulse 1s infinite',
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.1)' },
                    '100%': { transform: 'scale(1)' },
                  },
                }}
              />
            )}
          </IconButton>
        </span>
      </Tooltip>

      {/* Recording indicator */}
      {isRecording && (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            px: 2,
            py: 1,
            bgcolor: theme.palette.error.main,
            color: theme.palette.error.contrastText,
            borderRadius: 2,
            position: 'absolute',
            bottom: 60,
            left: 16,
            zIndex: 1000,
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              bgcolor: 'currentColor',
              borderRadius: '50%',
              animation: 'blink 1s infinite',
              '@keyframes blink': {
                '0%, 50%': { opacity: 1 },
                '51%, 100%': { opacity: 0.3 },
              },
            }}
          />
          <Typography variant="caption" fontWeight="bold">
            Recording {formatTime(Math.floor(recordingTime / 10))}
          </Typography>
          <VolumeUp sx={{ fontSize: 16, opacity: 0.5 + (audioLevel * 0.5) }} />
        </Box>
      )}

      {/* Transcript Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Voice Transcript</DialogTitle>
        <DialogContent>
          <Box
            sx={{
              p: 2,
              bgcolor: theme.palette.background.default,
              borderRadius: 1,
              minHeight: 100,
              border: 1,
              borderColor: 'divider',
            }}
          >
            <Typography variant="body2" sx={{ fontStyle: transcript ? 'normal' : 'italic' }}>
              {transcript || 'No speech detected. Please try again.'}
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleSendTranscript} 
            variant="contained"
            startIcon={<Send />}
            disabled={!transcript.trim()}
          >
            Send
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default VoiceRecorder;