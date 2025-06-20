
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Toaster } from "@/components/ui/toaster";
import Navigation from "@/components/Navigation";
import HomePage from "@/components/pages/HomePage";
import ScannerPage from "@/components/pages/ScannerPage";
import AboutPage from "@/components/pages/AboutPage";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/scanner" element={<ScannerPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
        <Toaster />
      </div>
    </Router>
  );
}

export default App;
