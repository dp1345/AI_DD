import express from "express"
import cors from "cors"
import path from "path"
import { fileURLToPath } from "url"
import multer from "multer"
import fs from "fs"

// ES module __dirname equivalent
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const PORT = process.env.PORT || 5000

// Middleware
app.use(cors())
app.use(express.json())

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "uploads")
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true })
    }
    cb(null, uploadDir)
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname)
  },
})

const upload = multer({
  storage: storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
  fileFilter: (req, file, cb) => {
    // Accept only video files
    if (file.mimetype.startsWith("video/")) {
      cb(null, true)
    } else {
      cb(new Error("Only video files are allowed!"), false)
    }
  },
})

// API Routes
app.post("/api/scan", upload.single("video"), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No video file uploaded" })
    }

    // In a real application, you would process the video here
    // For this example, we'll simulate a response after a delay
    setTimeout(() => {
      res.json({
        success: true,
        filename: req.file.filename,
        result: {
          isDeepfake: Math.random() > 0.5, // Random result for demo
          confidence: Math.floor(Math.random() * 100),
          analysisTime: Math.floor(Math.random() * 10) + 5,
        },
      })
    }, 3000)
  } catch (error) {
    console.error("Error processing upload:", error)
    res.status(500).json({ error: "Server error processing the video" })
  }
})

// Serve static files in production
if (process.env.NODE_ENV === "production") {
  app.use(express.static(path.join(__dirname, "../client/build")))

  app.get("*", (req, res) => {
    res.sendFile(path.join(__dirname, "../client/build", "index.html"))
  })
}

// Serve placeholder images for development
app.use("/placeholder.jpg", express.static(path.join(__dirname, "public/placeholder.jpg")))

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})

export default app

