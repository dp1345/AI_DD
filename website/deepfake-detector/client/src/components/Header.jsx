"use client"

import { useState } from "react"
import { Link } from "react-router-dom"

const Header = () => {
    const [isMenuOpen, setIsMenuOpen] = useState(false)

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen)
    }

    return (
        <header className="bg-blue-600 text-white shadow-md">
            <div className="container mx-auto px-4 py-4">
                <div className="flex justify-between items-center">
                    <Link to="/" className="text-2xl font-bold">
                        DeepfakeGuard
                    </Link>

                    {/* Mobile menu button */}
                    <button className="md:hidden focus:outline-none" onClick={toggleMenu} aria-label="Toggle menu">
                        <svg
                            className="w-6 h-6"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            {isMenuOpen ? (
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            ) : (
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            )}
                        </svg>
                    </button>

                    {/* Desktop navigation */}
                    <nav className="hidden md:flex space-x-6">
                        <Link to="/" className="hover:text-blue-200 transition-colors">
                            Home
                        </Link>
                        <Link to="/about" className="hover:text-blue-200 transition-colors">
                            About
                        </Link>
                        <Link to="/contact" className="hover:text-blue-200 transition-colors">
                            Contact
                        </Link>
                    </nav>
                </div>

                {/* Mobile navigation */}
                {isMenuOpen && (
                    <nav className="mt-4 md:hidden">
                        <ul className="flex flex-col space-y-2">
                            <li>
                                <Link to="/" className="block py-2 hover:text-blue-200 transition-colors">
                                    Home
                                </Link>
                            </li>
                            <li>
                                <Link to="/about" className="block py-2 hover:text-blue-200 transition-colors">
                                    About
                                </Link>
                            </li>
                            <li>
                                <Link to="/contact" className="block py-2 hover:text-blue-200 transition-colors">
                                    Contact
                                </Link>
                            </li>
                        </ul>
                    </nav>
                )}
            </div>
        </header>
    )
}

export default Header

