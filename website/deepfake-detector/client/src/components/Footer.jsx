const Footer = () => {
    const currentYear = new Date().getFullYear()

    return (
        <footer className="bg-gray-800 text-white py-6">
            <div className="container mx-auto px-4 text-center">
                <p>&copy; {currentYear} DeepfakeGuard. All rights reserved.</p>
                <div className="mt-2">
                    <a href="/privacy" className="text-blue-300 hover:text-blue-100 mx-2">
                        Privacy Policy
                    </a>
                    <a href="/terms" className="text-blue-300 hover:text-blue-100 mx-2">
                        Terms of Service
                    </a>
                </div>
            </div>
        </footer>
    )
}

export default Footer

