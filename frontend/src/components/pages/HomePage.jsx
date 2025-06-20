
import React from "react";
import { motion } from "framer-motion";

const HomePage = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <motion.div
      className="container mx-auto px-4 py-12"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      <motion.div className="text-center" variants={itemVariants}>
        <h1 className="text-5xl font-bold mb-6 text-gradient">Deepfake Detection</h1>
        <p className="text-xl mb-8">College Project on Deepfake Face Detection</p>
      </motion.div>

      <div className="max-w-4xl mx-auto space-y-8">
        <motion.div
          className="bg-white p-8 rounded-lg shadow-lg mb-8 card-hover"
          variants={itemVariants}
        >
          <h2 className="text-3xl font-bold mb-6">Team Members</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <motion.div
              className="p-4 rounded-lg bg-gradient-to-b from-blue-50 to-white border border-blue-100"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-xl font-semibold text-blue-700">Dhrumi Patel</h3>
            </motion.div>
            <motion.div
              className="p-4 rounded-lg bg-gradient-to-b from-purple-50 to-white border border-purple-100"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-xl font-semibold text-purple-700">Ayushi Depani</h3>
            </motion.div>
            <motion.div
              className="p-4 rounded-lg bg-gradient-to-b from-green-50 to-white border border-green-100"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-xl font-semibold text-green-700">Hiral Chauhan</h3>
            </motion.div>
            <motion.div
              className="p-4 rounded-lg bg-gradient-to-b from-red-50 to-white border border-red-100"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-xl font-semibold text-red-700">Roshani Patil</h3>
            </motion.div>
            <motion.div
              className="p-4 rounded-lg bg-gradient-to-b from-yellow-50 to-white border border-yellow-100"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-xl font-semibold text-yellow-700">Neha Phirke</h3>
            </motion.div>
          </div>
        </motion.div>

        <motion.div
          className="bg-white p-8 rounded-lg shadow-lg mb-8 card-hover"
          variants={itemVariants}
        >
          <h2 className="text-3xl font-bold mb-6">Pretrained Models Comparison</h2>
          <div className="grid gap-8 md:grid-cols-3">
            <motion.div
              className="p-6 rounded-lg bg-gradient-to-b from-blue-50 to-white border border-blue-100 card-hover"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-2xl font-semibold mb-4">EfficientNetB0</h3>
              <ul className="text-left text-lg space-y-2">
                <li>• Optimized architecture</li>
                <li>• Balanced performance</li>
                <li>• Efficient processing</li>
              </ul>
            </motion.div>

            <motion.div
              className="p-6 rounded-lg bg-gradient-to-b from-purple-50 to-white border border-purple-100 card-hover"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-2xl font-semibold mb-4">ConvexTiny</h3>
              <ul className="text-left text-lg space-y-2">
                <li>• Lightweight design</li>
                <li>• Fast inference</li>
                <li>• Resource-efficient</li>
              </ul>
            </motion.div>

            <motion.div
              className="p-6 rounded-lg bg-gradient-to-b from-green-50 to-white border border-green-100 card-hover"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <h3 className="text-2xl font-semibold mb-4">Xception</h3>
              <ul className="text-left text-lg space-y-2">
                <li>• Deep architecture</li>
                <li>• High accuracy</li>
                <li>• Feature extraction</li>
              </ul>
            </motion.div>
          </div>
        </motion.div>

        <motion.div
          className="bg-white p-8 rounded-lg shadow-lg card-hover"
          variants={itemVariants}
        >
          <h2 className="text-3xl font-bold mb-6">Dataset Information</h2>
          <div className="text-lg">
            <p className="mb-4">
              This project utilizes the <span className="font-semibold">Celeb-DF v1</span> dataset,
              which contains high-quality deepfake videos of celebrities.
            </p>
            <p>
              The dataset provides a comprehensive collection of real and manipulated facial content,
              enabling robust model training and evaluation.
            </p>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default HomePage;
