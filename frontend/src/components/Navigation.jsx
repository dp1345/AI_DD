
import React from "react";
import { Link } from "react-router-dom";
import { Home, Scan as Scanner, Info } from 'lucide-react';
import { motion } from "framer-motion";

const Navigation = () => {
  const navVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <motion.nav
      className="bg-white shadow-lg sticky top-0 z-50"
      initial="hidden"
      animate="visible"
      variants={navVariants}
    >
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="text-2xl font-bold text-gradient">
            DeepFake Detector
          </Link>
          <div className="flex space-x-8">
            <Link
              to="/"
              className="flex items-center text-lg hover:text-blue-600 transition-colors nav-link"
            >
              <Home className="mr-2" /> Home
            </Link>
            <Link
              to="/scanner"
              className="flex items-center text-lg hover:text-blue-600 transition-colors nav-link"
            >
              <Scanner className="mr-2" /> Scanner
            </Link>
            <Link
              to="/about"
              className="flex items-center text-lg hover:text-blue-600 transition-colors nav-link"
            >
              <Info className="mr-2" /> About
            </Link>
          </div>
        </div>
      </div>
    </motion.nav>
  );
};

export default Navigation;
