"use client"

import { useState } from "react"

const DeepfakeScanner = () => {
    const [file, setFile] = useState(null)
    const [url, setUrl] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState("")

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0]
        setFile(selectedFile)
        setUrl("")
        setResult(null)
        setError("")
    }

    const handleUrlChange = (e) => {
        setUrl(e.target.value)
        setFile(null)
        setResult(null)
        setError("")
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!file && !url) {
            setError("Please upload a file or enter a URL")
            return
        }

        setIsLoading(true)
        setError("")

        //     try {
        //         // Simulate API call with a timeout
        //         await new Promise((resolve) => setTimeout(resolve, 2000))

        //         // Mock result - in a real app, this would come from your API
        //         setResult({
        //             isDeepfake: Math.random() > 0.5,
        //             confidence: Math.floor(Math.random() * 100),
        //             analysisTime: (Math.random() * 2 + 0.5).toFixed(2),
        //         })
        //     } catch (err) {
        //         setError("An error occurred during analysis. Please try again.")
        //         console.error(err)
        //     } finally {
        //         setIsLoading(false)
        //     }
    }

    return (
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Deepfake Detection Tool</h2>

            <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-4">
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                        <input
                            type="file"
                            id="file-upload"
                            accept="image/*,video/*"
                            onChange={handleFileChange}
                            className="hidden"
                        />
                        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center justify-center">
                            <svg
                                className="w-12 h-12 text-gray-400 mb-3"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                                />
                            </svg>
                            <span className="text-gray-600 font-medium">
                                {file ? file.name : "Click to upload an  video"}
                            </span>
                            <span className="text-sm text-gray-500 mt-1">or drag and drop</span>
                        </label>
                    </div>

                    <div className="flex items-center">
                        <div className="flex-grow border-t border-gray-300"></div>
                        <span className="flex-shrink mx-4 text-gray-500">OR</span>
                        <div className="flex-grow border-t border-gray-300"></div>
                    </div>

                    <div>
                        <label htmlFor="url-input" className="block text-sm font-medium text-gray-700 mb-1">
                            Enter URL
                        </label>
                        <input
                            type="url"
                            id="url-input"
                            value={url}
                            onChange={handleUrlChange}
                            placeholder="https://example.com/video"
                            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                        />
                    </div>
                </div>

                {error && <div className="text-red-500 text-center">{error}</div>}

                <div className="text-center">
                    <button
                        type="submit"
                        disabled={isLoading || (!file && !url)}
                        className={`px-6 py-3 rounded-md text-white font-medium ${isLoading || (!file && !url)
                            ? "bg-gray-400 cursor-not-allowed"
                            : "bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                            }`}
                    >
                        {isLoading ? (
                            <span className="flex items-center justify-center">
                                <svg
                                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                >
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path
                                        className="opacity-75"
                                        fill="currentColor"
                                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    ></path>
                                </svg>
                                Analyzing...
                            </span>
                        ) : (
                            "Analyze"
                        )}
                    </button>
                </div>
            </form>

            {/* {result && (
                <div
                    className={`mt-8 p-6 rounded-lg ${result.isDeepfake ? "bg-red-50 border border-red-200" : "bg-green-50 border border-green-200"
                        }`}
                >
                    <h3 className={`text-xl font-bold mb-4 ${result.isDeepfake ? "text-red-700" : "text-green-700"}`}>
                        {result.isDeepfake ? "Potential Deepfake Detected" : "No Deepfake Detected"}
                    </h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-700">Confidence:</span>
                            <span className="font-medium">{result.confidence}%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-700">Analysis Time:</span>
                            <span className="font-medium">{result.analysisTime} seconds</span>
                        </div>
                        <div className="mt-4">
                            <p className={`text-sm ${result.isDeepfake ? "text-red-600" : "text-green-600"}`}>
                                {result.isDeepfake
                                    ? "This content shows signs of manipulation. We recommend verifying its authenticity from other sources."
                                    : "This content appears to be authentic based on our analysis."}
                            </p>
                        </div> 
                     </div>
                </div>
            )} */}
        </div>
    )
}

export default DeepfakeScanner

