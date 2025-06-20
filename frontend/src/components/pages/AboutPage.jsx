
import React from "react";
import { motion } from "framer-motion";

const AboutPage = () => {
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
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
      <motion.div className="max-w-3xl mx-auto" variants={containerVariants}>
        <motion.h1
          className="text-4xl font-bold mb-8 text-center"
          variants={itemVariants}
        >
          About Our Project
        </motion.h1>
        <motion.div
          className="bg-white p-8 rounded-lg shadow-lg card-hover"
          variants={itemVariants}
          whileHover={{ scale: 1.01 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <motion.p className="text-xl mb-6" variants={itemVariants}>
            This college project aims to combat the rising threat of deepfake technology by providing
            a reliable detection tool.
          </motion.p>
          <motion.h2 className="text-2xl font-semibold mb-4" variants={itemVariants}>
            Our Mission
          </motion.h2>
          <motion.p className="text-xl mb-6" variants={itemVariants}>
            To create a user-friendly platform that helps people identify manipulated media and
            protect themselves from digital deception.
          </motion.p>
          <motion.h2 className="text-2xl font-semibold mb-4" variants={itemVariants}>
            Technology
          </motion.h2>
          <motion.p className="text-xl" variants={itemVariants}>
            We use advanced machine learning algorithms and image processing techniques to analyze
            and detect.
          </motion.p>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default AboutPage;
