import React, { useState } from "react";
import { motion } from "framer-motion";
import { Upload, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import AnimatedCircle from "@/components/ui/AnimatedCircle"; // Import AnimatedCircle

const ScannerPage = () => {
  const { toast } = useToast();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [error, setError] = useState("");
  const [processing, setProcessing] = useState(false); // Processing state

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const validateFile = (file) => {
    if (!file) return false;

    const validTypes = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"];
    if (!validTypes.includes(file.type)) {
      toast({
        title: "Invalid file type",
        description: "Please upload a video file (MP4, MOV, AVI, or WebM)",
        variant: "destructive",
      });
      return false;
    }

    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      toast({
        title: "File too large",
        description: "Video size should be less than 100MB",
        variant: "destructive",
      });
      return false;
    }

    return true;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files[0];
    if (validateFile(file)) {
      setSelectedFile(file);
      setPredictions({});
      setError("");
      toast({
        title: "Video uploaded successfully",
        description: `Selected: ${file.name}`,
      });
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (validateFile(file)) {
      setSelectedFile(file);
      setPredictions({});
      setError("");
      toast({
        title: "Video uploaded successfully",
        description: `Selected: ${file.name}`,
      });
    }
  };

  const handleAnalyzeVideo = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select a video file before analyzing",
        variant: "destructive",
      });
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setProcessing(true); // Start spinner
    setError("");
    setPredictions({});

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData.error || "An error occurred while analyzing the video");
        toast({
          title: "Error",
          description: errorData.error || "An error occurred while analyzing the video",
          variant: "destructive",
        });
        return;
      }

      const result = await response.json();
      setPredictions(result.predictions);
      toast({
        title: "Analysis Complete",
        description: "Predictions are available below.",
      });
    } catch (error) {
      setError("Unable to connect to backend");
      toast({
        title: "Error",
        description: "Unable to connect to backend",
        variant: "destructive",
      });
    } finally {
      setProcessing(false); // Stop spinner
    }
  };

  return (
    <motion.div
      className="container mx-auto px-4 py-12"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div
        className="max-w-2xl mx-auto"
        initial={{ y: 20 }}
        animate={{ y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        <h1 className="text-4xl font-bold mb-8 text-center">Video Authenticity Detector</h1>
        <motion.div
          className="bg-white p-8 rounded-lg shadow-lg card-hover"
          whileHover={{ scale: 1.01 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <div
            className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300
              ${dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"}
              ${selectedFile ? "border-green-500 bg-green-50" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="video/*"
              onChange={handleFileInput}
              className="hidden"
              id="file-upload"
            />

            <motion.div
              className="space-y-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              {selectedFile ? (
                <motion.div
                  className="flex flex-col items-center space-y-4"
                  initial={{ scale: 0.9 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 200 }}
                >
                  <Video className="w-16 h-16 text-green-500" />
                  <p className="text-xl font-medium text-green-700">{selectedFile.name}</p>
                  <p className="text-sm text-green-600">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </motion.div>
              ) : (
                <>
                  <Upload className="w-16 h-16 mx-auto text-gray-400" />
                  <p className="text-xl mb-2">Drag and drop your video here</p>
                  <p className="text-lg text-gray-500 mb-4">or</p>
                </>
              )}
              {!selectedFile && (
                <p className="text-sm text-gray-500 mt-4">
                  Supports: MP4, MOV, AVI, WebM (Max 100MB)
                </p>
              )}
            </motion.div>
          </div>

          <div className="flex flex-col space-y-4 mt-6">
            <Button
              onClick={() => document.getElementById("file-upload").click()}
              variant="outline"
              className="w-full"
            >
              {selectedFile ? "Choose Different Video" : "Browse Files"}
            </Button>

            {selectedFile && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <Button
                  className="w-full bg-blue-600 hover:bg-blue-700"
                  onClick={handleAnalyzeVideo}
                >
                  Analyze Video
                </Button>
              </motion.div>
            )}
          </div>

          {processing && (
            <div className="flex flex-col items-center space-y-4 mt-4">
              <AnimatedCircle /> {/* Use the updated AnimatedCircle */}
              <p className="text-blue-500 text-sm mt-2">Video is under process...</p>
            </div>
          )}

          {error && <p className="text-red-500 mt-4">{error}</p>}
          {Object.keys(predictions).length > 0 && (
            <div className="mt-4">
              <h2 className="text-lg font-medium">Predictions:</h2>
              <ul className="list-disc list-inside">
                {Object.entries(predictions).map(([modelName, result]) => (
                  <li key={modelName}>
                    <strong>{modelName}:</strong> {result.prediction}
                    <span className="text-gray-600 ml-2">
                      (Confidence: {(result.confidence * 100).toFixed(2)}%)
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default ScannerPage;