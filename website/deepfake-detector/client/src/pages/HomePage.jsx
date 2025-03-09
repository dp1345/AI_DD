import DeepfakeScanner from "../components/DeepfakeScanner"
import ThreatInfo from "../components/ThreatInfo"

const HomePage = () => {
    return (
        <div className="container mx-auto px-4 py-8">
            <DeepfakeScanner />
            <ThreatInfo />
        </div>
    )
}

export default HomePage

