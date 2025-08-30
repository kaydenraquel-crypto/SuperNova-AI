import React, { useState, useRef, useCallback } from 'react';
import {
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
} from '@mui/material';
import {
  AttachFile,
  CloudUpload,
  Description,
  Image,
  PictureAsPdf,
  DeleteOutline,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface UploadedFile {
  id: string;
  file: File;
  status: 'uploading' | 'completed' | 'error';
  progress: number;
  url?: string;
  error?: string;
}

interface FileUploaderProps {
  onFileUpload: (files: File[]) => void;
  disabled?: boolean;
  acceptedTypes?: string[];
  maxFileSize?: number; // in MB
  maxFiles?: number;
}

const FileUploader: React.FC<FileUploaderProps> = ({
  onFileUpload,
  disabled = false,
  acceptedTypes = ['.pdf', '.png', '.jpg', '.jpeg', '.txt', '.csv', '.xlsx'],
  maxFileSize = 10,
  maxFiles = 5,
}) => {
  const theme = useTheme();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragOver, setDragOver] = useState(false);

  const getFileIcon = (file: File) => {
    const type = file.type;
    const name = file.name.toLowerCase();
    
    if (type.startsWith('image/')) return <Image />;
    if (type === 'application/pdf' || name.endsWith('.pdf')) return <PictureAsPdf />;
    return <Description />;
  };

  const getFileTypeColor = (file: File) => {
    const type = file.type;
    const name = file.name.toLowerCase();
    
    if (type.startsWith('image/')) return 'success';
    if (type === 'application/pdf' || name.endsWith('.pdf')) return 'error';
    if (name.endsWith('.csv') || name.endsWith('.xlsx')) return 'info';
    return 'default';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const validateFile = (file: File): string | null => {
    if (file.size > maxFileSize * 1024 * 1024) {
      return `File size exceeds ${maxFileSize}MB limit`;
    }
    
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(extension)) {
      return `File type not supported. Accepted types: ${acceptedTypes.join(', ')}`;
    }
    
    return null;
  };

  const simulateUpload = (file: UploadedFile) => {
    const interval = setInterval(() => {
      setUploadedFiles(prev => 
        prev.map(f => {
          if (f.id === file.id) {
            const newProgress = Math.min(f.progress + Math.random() * 20, 100);
            if (newProgress >= 100) {
              clearInterval(interval);
              return { 
                ...f, 
                progress: 100, 
                status: 'completed', 
                url: URL.createObjectURL(f.file) 
              };
            }
            return { ...f, progress: newProgress };
          }
          return f;
        })
      );
    }, 200);
  };

  const handleFileSelect = useCallback((files: FileList) => {
    const fileArray = Array.from(files);
    
    if (uploadedFiles.length + fileArray.length > maxFiles) {
      alert(`Maximum ${maxFiles} files allowed`);
      return;
    }

    const newFiles: UploadedFile[] = fileArray.map(file => {
      const error = validateFile(file);
      return {
        id: Math.random().toString(36).substr(2, 9),
        file,
        status: error ? 'error' : 'uploading',
        progress: error ? 0 : 10,
        error,
      };
    });

    setUploadedFiles(prev => [...prev, ...newFiles]);
    
    // Start upload simulation for valid files
    newFiles.forEach(file => {
      if (!file.error) {
        simulateUpload(file);
      }
    });

    setDialogOpen(true);
  }, [uploadedFiles.length, maxFiles]);

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      handleFileSelect(files);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    const files = event.dataTransfer.files;
    if (files) {
      handleFileSelect(files);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id));
  };

  const handleSendFiles = () => {
    const completedFiles = uploadedFiles
      .filter(f => f.status === 'completed')
      .map(f => f.file);
    
    if (completedFiles.length > 0) {
      onFileUpload(completedFiles);
      setUploadedFiles([]);
      setDialogOpen(false);
    }
  };

  const completedCount = uploadedFiles.filter(f => f.status === 'completed').length;

  return (
    <>
      <Tooltip title="Upload files">
        <span>
          <IconButton
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            sx={{
              '&:hover': {
                bgcolor: `${theme.palette.primary.main}10`,
              },
            }}
          >
            <AttachFile />
          </IconButton>
        </span>
      </Tooltip>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        multiple
        accept={acceptedTypes.join(',')}
        style={{ display: 'none' }}
      />

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>File Upload</DialogTitle>
        <DialogContent>
          {/* Drop Zone */}
          <Box
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            sx={{
              border: 2,
              borderStyle: 'dashed',
              borderColor: dragOver ? theme.palette.primary.main : 'divider',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              bgcolor: dragOver ? `${theme.palette.primary.main}05` : 'transparent',
              mb: 2,
              transition: 'all 0.2s ease-in-out',
            }}
          >
            <CloudUpload 
              sx={{ 
                fontSize: 48, 
                color: dragOver ? theme.palette.primary.main : 'text.secondary',
                mb: 1,
              }} 
            />
            <Typography variant="h6" gutterBottom>
              Drop files here or click to browse
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Supported formats: {acceptedTypes.join(', ')}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Max file size: {maxFileSize}MB • Max files: {maxFiles}
            </Typography>
          </Box>

          {/* File List */}
          {uploadedFiles.length > 0 && (
            <List>
              {uploadedFiles.map((file) => (
                <ListItem key={file.id} divider>
                  <ListItemIcon>
                    {file.status === 'completed' ? (
                      <CheckCircle color="success" />
                    ) : file.status === 'error' ? (
                      <Error color="error" />
                    ) : (
                      getFileIcon(file.file)
                    )}
                  </ListItemIcon>
                  
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" noWrap>
                          {file.file.name}
                        </Typography>
                        <Chip
                          label={formatFileSize(file.file.size)}
                          size="small"
                          color={getFileTypeColor(file.file) as any}
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        {file.status === 'uploading' && (
                          <LinearProgress 
                            variant="determinate" 
                            value={file.progress}
                            sx={{ mb: 1 }}
                          />
                        )}
                        {file.error && (
                          <Typography variant="caption" color="error">
                            {file.error}
                          </Typography>
                        )}
                        {file.status === 'completed' && (
                          <Typography variant="caption" color="success.main">
                            Upload complete
                          </Typography>
                        )}
                        {file.status === 'uploading' && (
                          <Typography variant="caption" color="text.secondary">
                            Uploading... {Math.round(file.progress)}%
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={() => removeFile(file.id)}>
                      <DeleteOutline />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          <Button onClick={() => fileInputRef.current?.click()}>
            Add More Files
          </Button>
          <Button
            onClick={handleSendFiles}
            variant="contained"
            disabled={completedCount === 0}
          >
            Send Files ({completedCount})
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default FileUploader;