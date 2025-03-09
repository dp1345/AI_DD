const ThreatInfo = () => {
    const threats = [
        {
            title: "Identity Theft",
            description: "Deepfakes can be used to impersonate individuals, potentially leading to identity theft and fraud.",
            icon: "user-x",
        },
        {
            title: "Misinformation",
            description: "Synthetic media can spread false information rapidly, undermining trust in authentic content.",
            icon: "alert-triangle",
        },
        {
            title: "Reputation Damage",
            description:
                "Individuals may suffer severe reputation damage from convincing deepfakes depicting them in compromising situations.",
            icon: "shield-off",
        },
        {
            title: "Financial Fraud",
            description:
                "Voice cloning and video deepfakes have been used to commit financial fraud by impersonating executives or trusted figures.",
            icon: "dollar-sign",
        },
    ]

    const protectionTips = [
        "Verify content through multiple trusted sources",
        "Check for visual inconsistencies in suspected deepfakes",
        "Be skeptical of sensational content, especially if it lacks context",
        "Use deepfake detection tools like this one",
        "Report suspected deepfakes to the platform where they appear",
    ]

    return (
        <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Understanding Deepfake Threats</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {threats.map((threat, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow">
                        <div className="flex items-start">
                            <div className="mr-4 bg-blue-100 p-3 rounded-full">
                                <svg
                                    className="w-6 h-6 text-blue-600"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                                    />
                                </svg>
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-gray-800 mb-2">{threat.title}</h3>
                                <p className="text-gray-600">{threat.description}</p>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <div className="bg-blue-50 rounded-lg p-6 border border-blue-100">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">How to Protect Yourself</h3>
                <ul className="space-y-3">
                    {protectionTips.map((tip, index) => (
                        <li key={index} className="flex items-start">
                            <svg
                                className="w-5 h-5 text-blue-600 mr-2 mt-0.5"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <span className="text-gray-700">{tip}</span>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="mt-8 text-center">
                <p className="text-gray-600">
                    Stay informed about deepfake technology and detection methods to better protect yourself and others.
                </p>
                <a
                    href="https://www.ftc.gov/business-guidance/blog/2020/02/deep-fakes-what-are-they-why-care"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-block mt-2 text-blue-600 hover:text-blue-800 hover:underline"
                >
                    Learn more about deepfakes from the FTC
                </a>
            </div>
        </div>
    )
}

export default ThreatInfo

