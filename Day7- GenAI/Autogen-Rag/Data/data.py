urban_ag_data = {
    "vertical_farming": {
        "description": "Agricultural practice involving growing crops in vertically stacked layers",
        "suitable_crops": ["lettuce", "spinach", "herbs", "microgreens", "strawberries"],
        "space_efficiency": 0.9,
        "water_usage": 0.1,
        "energy_requirement": 0.8,
        "advantages": [
            "Maximizes limited urban space",
            "Controlled environment for year-round growing",
            "Protection from pests and weather extremes",
            "Reduced water consumption through recirculation",
            "Can be implemented in unused urban buildings"
        ],
        "disadvantages": [
            "High initial setup costs",
            "Requires technical knowledge",
            "Energy-intensive for lighting and climate control",
            "Limited crop diversity (primarily leafy greens and herbs)",
            "Dependency on technology and electricity"
        ],
        "ideal_locations": [
            "Warehouses",
            "Abandoned buildings",
            "Basements",
            "Indoor spaces with climate control"
        ],
        "startup_cost_range": "$500-$3000 per square meter depending on technology level"
    },
    "rooftop_gardens": {
        "description": "Gardens established on roofs of buildings",
        "suitable_crops": ["tomatoes", "peppers", "eggplants", "herbs", "leafy greens"],
        "space_efficiency": 0.7,
        "water_usage": 0.5,
        "energy_requirement": 0.2,
        "advantages": [
            "Utilizes otherwise unused space",
            "Helps reduce urban heat island effect",
            "Improves building insulation",
            "Natural sunlight reduces energy costs",
            "Can support a wide variety of crops"
        ],
        "disadvantages": [
            "Weight limitations on some buildings",
            "Exposure to weather extremes",
            "Potential roof damage if improperly installed",
            "Limited to growing season without additional protection",
            "May require structural assessment before implementation"
        ],
        "ideal_locations": [
            "Flat commercial rooftops",
            "Residential buildings with accessible roofs",
            "Buildings with strong structural integrity",
            "Areas with good sun exposure"
        ],
        "startup_cost_range": "$150-$500 per square meter depending on infrastructure needs"
    },
    "hydroponic_systems": {
        "description": "Method of growing plants without soil, using mineral nutrient solutions in water",
        "suitable_crops": ["lettuce", "spinach", "tomatoes", "cucumbers", "herbs"],
        "space_efficiency": 0.95,
        "water_usage": 0.05,
        "energy_requirement": 0.7,
        "advantages": [
            "Extremely water efficient (up to 95% less than traditional farming)",
            "Fast growth rates (30-50% faster than soil)",
            "No soil-borne diseases or pests",
            "Precise nutrient control",
            "Can be stacked vertically for additional space efficiency"
        ],
        "disadvantages": [
            "Requires careful monitoring of nutrient solution",
            "Vulnerable to power outages",
            "Initial setup costs can be high",
            "Requires technical knowledge",
            "Some systems need regular maintenance"
        ],
        "ideal_locations": [
            "Garages",
            "Basements",
            "Dedicated growing rooms",
            "Greenhouses",
            "Converted shipping containers"
        ],
        "startup_cost_range": "$300-$1000 per square meter depending on system type"
    },
    "container_gardening": {
        "description": "Growing plants in containers instead of planting them in the ground",
        "suitable_crops": ["herbs", "leafy greens", "radishes", "carrots", "bush beans"],
        "space_efficiency": 0.6,
        "water_usage": 0.6,
        "energy_requirement": 0.1,
        "advantages": [
            "Highly flexible and portable",
            "Low startup costs",
            "Accessible for beginners",
            "Can be done on balconies, patios, or windowsills",
            "Avoids contaminated urban soils"
        ],
        "disadvantages": [
            "Limited root space for plants",
            "Requires regular watering",
            "Limited yield compared to other methods",
            "Containers may deteriorate over time",
            "May need additional fertilization"
        ],
        "ideal_locations": [
            "Balconies",
            "Patios",
            "Window sills",
            "Small yards",
            "Indoor spaces with good light"
        ],
        "startup_cost_range": "$50-$200 per square meter depending on container quality"
    },
    "aquaponic_systems": {
        "description": "System combining aquaculture with hydroponics in a symbiotic environment",
        "suitable_crops": ["lettuce", "herbs", "kale", "swiss chard", "watercress"],
        "space_efficiency": 0.85,
        "water_usage": 0.1,
        "energy_requirement": 0.6,
        "advantages": [
            "Produces both plants and fish as food sources",
            "Self-fertilizing system (fish waste feeds plants)",
            "Uses 90-95% less water than traditional farming",
            "Minimal waste production",
            "Can be more productive per square foot than soil gardening"
        ],
        "disadvantages": [
            "Complex to set up and balance initially",
            "Requires monitoring of water chemistry",
            "Higher startup costs than some methods",
            "Energy required for pumps and filtration",
            "Vulnerable to system failures"
        ],
        "ideal_locations": [
            "Basements with good temperature control",
            "Garages",
            "Greenhouses",
            "Dedicated indoor spaces",
            "Areas that can support tank weight"
        ],
        "startup_cost_range": "$500-$1500 per square meter depending on system complexity"
    },
    "aeroponic_systems": {
        "description": "Soil-free growing system where plant roots are misted with nutrient solution",
        "suitable_crops": ["lettuce", "kale", "herbs", "strawberries", "bok choy"],
        "space_efficiency": 0.97,
        "water_usage": 0.03,
        "energy_requirement": 0.9,
        "advantages": [
            "Highest water efficiency of any growing method",
            "Faster growth rates than hydroponics",
            "Reduced disease transmission risk",
            "Mineral-rich mist enhances nutrient uptake",
            "Extremely space-efficient vertical configurations"
        ],
        "disadvantages": [
            "Extremely high initial investment costs",
            "Critical dependency on precise misting schedules",
            "Vulnerable to pump failures and clogged nozzles",
            "Requires advanced technical knowledge",
            "Limited root crop compatibility"
        ],
        "ideal_locations": [
            "High-tech research facilities",
            "Pharmaceutical green walls",
            "Underground growing bunkers",
            "Modular shipping container farms"
        ],
        "startup_cost_range": "$1200-$5000 per square meter depending on automation"
    },
    "mushroom_cultivation": {
        "description": "Growing fungi in controlled environments using organic substrates",
        "suitable_crops": ["oyster_mushrooms", "shiitake", "lion's_mane", "button_mushrooms"],
        "space_efficiency": 0.85,
        "water_usage": 0.7,
        "energy_requirement": 0.5,
        "advantages": [
            "Can utilize waste materials as growing substrate",
            "High-value specialty crop production",
            "Thrives in dark, unused urban spaces",
            "Short growth cycles (2-3 weeks)",
            "Low light requirements reduce energy costs"
        ],
        "disadvantages": [
            "Requires strict humidity and temperature control",
            "High contamination risk from mold/bacteria",
            "Labor-intensive substrate preparation",
            "Limited market channels for fresh mushrooms",
            "Specialized harvesting/storage requirements"
        ],
        "ideal_locations": [
            "Abandoned subway tunnels",
            "Basement grow rooms",
            "Temperature-controlled warehouses",
            "Repurposed industrial coolers"
        ],
        "startup_cost_range": "$300-$1000 per square meter depending on climate control needs"
    },
    "green_walls": {
        "description": "Vegetated vertical surfaces integrated with building structures",
        "suitable_crops": ["strawberries", "herbs", "leafy_greens", "succulents", "edible_flowers"],
        "space_efficiency": 0.92,
        "water_usage": 0.4,
        "energy_requirement": 0.3,
        "advantages": [
            "Improves building insulation and air quality",
            "Acts as natural sound barrier",
            "Aesthetic urban beautification",
            "Supports biodiversity with pollinator-friendly plants",
            "Can integrate with building water systems"
        ],
        "disadvantages": [
            "Requires specialized irrigation systems",
            "Potential structural stress on building walls",
            "Higher maintenance access challenges",
            "Limited root depth for larger plants",
            "Wind exposure issues at height"
        ],
        "ideal_locations": [
            "Building exteriors with southern exposure",
            "Interior atrium walls",
            "Noise barrier installations",
            "Retail space dividers",
            "Public transportation hubs"
        ],
        "startup_cost_range": "$800-$2500 per square meter depending on automation"
    },
    "microgreen_hubs": {
        "description": "High-density specialized systems for commercial microgreen production",
        "suitable_crops": ["broccoli_microgreens", "radish_microgreens", "pea_shoots", "sunflower_shoots", "wheatgrass"],
        "space_efficiency": 0.99,
        "water_usage": 0.08,
        "energy_requirement": 0.4,
        "advantages": [
            "Ultra-fast crop turnover (7-14 days)",
            "Extremely high nutritional density",
            "Low space requirements",
            "Year-round production capability",
            "High premium market prices"
        ],
        "disadvantages": [
            "Labor-intensive harvesting",
            "Requires sterile growing conditions",
            "Limited shelf life after harvest",
            "Niche market dependence",
            "Constant seed supply needed"
        ],
        "ideal_locations": [
            "Commercial kitchens",
            "Restaurant basements",
            "Grocery store backrooms",
            "Food truck commissaries",
            "Shared commercial incubators"
        ],
        "startup_cost_range": "$200-$600 per square meter depending on scale"
    },
    "raised_beds": {
        "description": "Elevated soil containers above ground level",
        "suitable_crops": ["carrots", "beets", "potatoes", "garlic", "berries"],
        "space_efficiency": 0.65,
        "water_usage": 0.55,
        "energy_requirement": 0.15,
        "advantages": [
            "Avoids contaminated soils",
            "Better drainage than ground planting",
            "Easier access for gardeners",
            "Warmer soil temperatures in spring",
            "Customizable soil composition"
        ],
        "disadvantages": [
            "Limited root depth for some crops",
            "Higher material costs than in-ground",
            "Soil compaction over time",
            "Requires periodic replenishment",
            "Not easily movable"
        ],
        "ideal_locations": [
            "Contaminated brownfields",
            "Parking lot conversions",
            "Community garden plots",
            "Schoolyards",
            "Vacant lots with poor soil"
        ],
        "startup_cost_range": "$100-$300 per square meter"
    },
    "greenhouse_growing": {
        "description": "Controlled environment agriculture using transparent structures",
        "suitable_crops": ["tomatoes", "peppers", "cucumbers", "citrus", "tropical_plants"],
        "space_efficiency": 0.75,
        "water_usage": 0.4,
        "energy_requirement": 0.65,
        "advantages": [
            "Year-round production",
            "Protection from extreme weather",
            "Higher humidity retention",
            "Pest exclusion capabilities",
            "Extended growing seasons"
        ],
        "disadvantages": [
            "High energy costs for heating/cooling",
            "Glass/polycarbonate maintenance",
            "Ventilation requirements",
            "Condensation management",
            "Limited natural pollination"
        ],
        "ideal_locations": [
            "Rooftops with structural capacity",
            "Vacant industrial land",
            "Brownfield sites",
            "Community garden centers",
            "Research institutions"
        ],
        "startup_cost_range": "$200-$800 per square meter"
    }
}

urban_crop_data = {
    "lettuce": {
        "growth_cycle_days": 30,
        "yield_per_sqft": 0.5,
        "water_needs": "low",
        "light_needs": "medium",
        "temperature_range": [45, 75],
        "difficulty": "easy",
        "market_value": "medium",
        "nutritional_highlights": "High in vitamins A, K, and folate",
        "best_growing_methods": ["hydroponic_systems", "vertical_farming", "container_gardening"],
        "companion_plants": ["radishes", "carrots", "strawberries"],
        "seasonal_notes": "Grows well in spring and fall; summer varieties need shade in hot climates"
    },
    "tomatoes": {
        "growth_cycle_days": 90,
        "yield_per_sqft": 0.8,
        "water_needs": "medium",
        "light_needs": "high",
        "temperature_range": [65, 85],
        "difficulty": "medium",
        "market_value": "high",
        "nutritional_highlights": "Rich in lycopene, vitamins C, K, and potassium",
        "best_growing_methods": ["rooftop_gardens", "container_gardening"],
        "companion_plants": ["basil", "marigolds", "carrots"],
        "seasonal_notes": "Summer crop; needs protection from extreme heat; determinate varieties better for containers"
    },
    "microgreens": {
        "growth_cycle_days": 14,
        "yield_per_sqft": 0.3,
        "water_needs": "low",
        "light_needs": "medium",
        "temperature_range": [60, 75],
        "difficulty": "easy",
        "market_value": "very high",
        "nutritional_highlights": "Up to 40 times more nutrients than mature plants",
        "best_growing_methods": ["vertical_farming", "container_gardening", "hydroponic_systems"],
        "companion_plants": ["Not applicable - typically grown alone"],
        "seasonal_notes": "Can be grown year-round indoors; quick turnover allows multiple harvests"
    },
    "herbs": {
        "growth_cycle_days": 45,
        "yield_per_sqft": 0.4,
        "water_needs": "low",
        "light_needs": "medium",
        "temperature_range": [55, 80],
        "difficulty": "easy",
        "market_value": "high",
        "nutritional_highlights": "Rich in essential oils and antioxidants",
        "best_growing_methods": ["container_gardening", "vertical_farming", "hydroponic_systems"],
        "companion_plants": ["Varies by herb type"],
        "seasonal_notes": "Many herbs can grow year-round indoors; perennial herbs offer multiple harvests"
    },
    "strawberries": {
        "growth_cycle_days": 90,
        "yield_per_sqft": 0.6,
        "water_needs": "medium",
        "light_needs": "high",
        "temperature_range": [60, 80],
        "difficulty": "medium",
        "market_value": "high",
        "nutritional_highlights": "High in vitamin C, manganese, and antioxidants",
        "best_growing_methods": ["vertical_farming", "container_gardening"],
        "companion_plants": ["lettuce", "spinach", "thyme"],
        "seasonal_notes": "Spring/summer crop; everbearing varieties provide multiple harvests"
    },
    "peppers": {
        "growth_cycle_days": 80,
        "yield_per_sqft": 0.7,
        "water_needs": "medium",
        "light_needs": "high",
        "temperature_range": [70, 90],
        "difficulty": "medium",
        "market_value": "medium",
        "nutritional_highlights": "High in vitamins A, C, and capsaicin (in hot varieties)",
        "best_growing_methods": ["rooftop_gardens", "container_gardening"],
        "companion_plants": ["basil", "onions", "carrots"],
        "seasonal_notes": "Summer crop; needs warm soil to germinate; smaller varieties better for containers"
    },
    "kale": {
        "growth_cycle_days": 55,
        "yield_per_sqft": 0.65,
        "water_needs": "medium",
        "light_needs": "high",
        "temperature_range": [45, 75],
        "difficulty": "medium",
        "market_value": "high",
        "nutritional_highlights": "Excellent source of vitamins K, A, C, and calcium",
        "best_growing_methods": ["vertical_farming", "aeroponic_systems", "container_gardening"],
        "companion_plants": ["beets", "nasturtiums", "garlic"],
        "seasonal_notes": "Cold-tolerant; improves flavor after frost exposure; prone to bolting in heat"
    },
    "oyster_mushrooms": {
        "growth_cycle_days": 21,
        "yield_per_sqft": 0.45,
        "water_needs": "high",
        "light_needs": "none",
        "temperature_range": [55, 65],
        "difficulty": "medium",
        "market_value": "very high",
        "nutritional_highlights": "Rich in protein, fiber, and antioxidants like ergothioneine",
        "best_growing_methods": ["mushroom_cultivation", "indoor_controlled_environments"],
        "companion_plants": ["Not applicable - grown in isolated substrates"],
        "seasonal_notes": "Year-round indoor production; requires 85-95% humidity during fruiting"
    },
    "arugula": {
        "growth_cycle_days": 40,
        "yield_per_sqft": 0.55,
        "water_needs": "medium",
        "light_needs": "medium",
        "temperature_range": [50, 75],
        "difficulty": "easy",
        "market_value": "high",
        "nutritional_highlights": "Rich in glucosinolates and vitamin K",
        "best_growing_methods": ["green_walls", "hydroponic_systems", "container_gardening"],
        "companion_plants": ["bush_beans", "nasturtiums", "dill"],
        "seasonal_notes": "Prefers cool weather; bolts quickly in heat - ideal for climate-controlled systems"
    },
    "bok_choy": {
        "growth_cycle_days": 45,
        "yield_per_sqft": 0.7,
        "water_needs": "high",
        "light_needs": "medium",
        "temperature_range": [55, 70],
        "difficulty": "medium",
        "market_value": "medium",
        "nutritional_highlights": "Excellent source of vitamins A, C, and calcium",
        "best_growing_methods": ["aeroponic_systems", "vertical_farming", "hydroponic_systems"],
        "companion_plants": ["carrots", "mint", "marigolds"],
        "seasonal_notes": "Best in spring/fall; requires consistent moisture for proper head formation"
    },
    "dwarf_citrus": {
        "growth_cycle_days": 365,
        "yield_per_sqft": 2.5,
        "water_needs": "medium",
        "light_needs": "high",
        "temperature_range": [60, 85],
        "difficulty": "hard",
        "market_value": "very high",
        "nutritional_highlights": "High in vitamin C and flavonoids",
        "best_growing_methods": ["container_gardening", "greenhouse_growing", "rooftop_gardens"],
        "companion_plants": ["basil", "lavender", "chives"],
        "seasonal_notes": "Evergreen but needs winter protection in cold climates; fruits ripen over 6-9 months"
    },
    "edible_flowers": {
        "growth_cycle_days": 60,
        "yield_per_sqft": 0.2,
        "water_needs": "low",
        "light_needs": "high",
        "temperature_range": [65, 80],
        "difficulty": "medium",
        "market_value": "premium",
        "nutritional_highlights": "Contain unique antioxidants and phytochemicals",
        "best_growing_methods": ["green_walls", "container_gardening", "hydroponic_systems"],
        "companion_plants": ["lettuce", "herbs", "radishes"],
        "seasonal_notes": "Require careful temperature control for optimal pigmentation development"
    },
    "shiitake": {
        "growth_cycle_days": 84,
        "yield_per_sqft": 0.35,
        "water_needs": "high",
        "light_needs": "low",
        "temperature_range": [50, 70],
        "difficulty": "hard",
        "market_value": "premium",
        "nutritional_highlights": "Contains lentinan with immune-boosting properties",
        "best_growing_methods": ["mushroom_cultivation"],
        "companion_plants": ["Not applicable"],
        "seasonal_notes": "Requires 2-phase growth process (incubation then fruiting)"
    },
    "pea_shoots": {
        "growth_cycle_days": 10,
        "yield_per_sqft": 0.4,
        "water_needs": "medium",
        "light_needs": "low",
        "temperature_range": [65, 75],
        "difficulty": "easy",
        "market_value": "very high",
        "nutritional_highlights": "High in vitamins C, A, and folate",
        "best_growing_methods": ["microgreen_hubs", "vertical_farming"],
        "companion_plants": ["Not applicable"],
        "seasonal_notes": "Year-round production in controlled environments"
    }
}

urban_challenges = {
    "limited_space": {
        "description": "Urban areas have limited growing space",
        "solutions": ["vertical farming", "container gardening", "rooftop gardens"],
        "impact_score": 9,
        "detailed_strategies": [
            "Utilize wall space with vertical growing systems",
            "Grow upward with trellises for vining crops",
            "Use hanging planters for small crops",
            "Choose compact and dwarf varieties",
            "Practice succession planting to maximize harvest per square foot"
        ],
        "recommended_resources": [
            "Space-Saving Garden Plans by Garden Design Magazine",
            "Vertical Gardening by Derek Fell",
            "Square Foot Gardening by Mel Bartholomew"
        ]
    },
    "soil_contamination": {
        "description": "Urban soils often contain pollutants and heavy metals",
        "solutions": ["hydroponic systems", "raised beds", "container gardening"],
        "impact_score": 8,
        "detailed_strategies": [
            "Test soil before planting directly in ground",
            "Use food-grade containers for growing edibles",
            "Create raised beds with barriers between existing soil",
            "Import clean growing medium from verified sources",
            "Consider soil-free growing methods like hydroponics"
        ],
        "recommended_resources": [
            "EPA guidelines for urban gardening",
            "Urban Soil Lead Remediation guide by Cornell University",
            "Soilless Growing Techniques for Urban Farmers"
        ]
    },
    "water_access": {
        "description": "Limited access to clean water for irrigation",
        "solutions": ["rainwater harvesting", "hydroponic systems", "drip irrigation"],
        "impact_score": 7,
        "detailed_strategies": [
            "Install rain barrels to collect roof runoff",
            "Implement drip irrigation systems for water efficiency",
            "Use mulch to retain soil moisture",
            "Select drought-tolerant crop varieties",
            "Recycle greywater when appropriate and legal"
        ],
        "recommended_resources": [
            "Urban Rainwater Harvesting Systems guide",
            "Water-Wise Gardening for Urban Spaces",
            "DIY Drip Irrigation Systems for Small Spaces"
        ]
    },
    "light_limitations": {
        "description": "Buildings creating shadows and reduced sunlight",
        "solutions": ["grow lights", "reflective surfaces", "strategic crop placement"],
        "impact_score": 8,
        "detailed_strategies": [
            "Map sun patterns throughout the day and year",
            "Use reflective surfaces to direct light to plants",
            "Install grow lights to supplement natural light",
            "Choose shade-tolerant crops for low-light areas",
            "Use mobile containers to follow sunlight patterns"
        ],
        "recommended_resources": [
            "Shade-Tolerant Vegetables guide by Urban Farmer",
            "LED Grow Light Selection for Indoor Growing",
            "Light Mapping Techniques for Urban Gardens"
        ]
    },
    "climate_variability": {
        "description": "Urban heat islands and microclimate effects",
        "solutions": ["greenhouse growing", "climate-controlled environments", "seasonal planning","green_walls", "aeroponic_systems"],
        "impact_score": 6,
        "detailed_strategies": [
            "Use cold frames to extend growing seasons",
            "Create microclimates with strategic windbreaks",
            "Install shade cloth during extreme heat periods",
            "Use row covers for frost protection",
            "Monitor urban heat island effects and adjust planning"
        ],
        "recommended_resources": [
            "Urban Microclimate Mapping Tools",
            "Season Extension Techniques for City Growers",
            "Climate-Resilient Urban Agriculture guide"
        ]
    },
    "zoning_regulations": {
        "description": "Municipal codes restricting agricultural activities in urban areas",
        "solutions": ["vertical_farming", "indoor_growing", "policy_advocacy"],
        "impact_score": 7,
        "detailed_strategies": [
            "Lobby for urban agriculture zoning amendments",
            "Utilize rooftop spaces exempt from ground-level restrictions",
            "Implement hidden growing systems in allowable structures",
            "Partner with educational institutions for protected status",
            "Advocate for 'green infrastructure' classifications"
        ],
        "recommended_resources": [
            "Urban Agriculture Zoning Practice Guide (APA)",
            "Model Urban Growers Ordinance Toolkit",
            "Municipal Code Hackers Handbook for Urban Farmers"
        ]
    },
    "air_pollution": {
        "description": "Particulate matter and pollutants affecting crop safety",
        "solutions": ["green_walls", "indoor_growing", "air_filtering_plants", "microgreen_hubs"],
        "impact_score": 7,
        "detailed_strategies": [
            "Install HEPA filtration in indoor grow spaces",
            "Use hairy-leaf plants as natural air filters",
            "Implement positive air pressure systems",
            "Test crops for heavy metal absorption",
            "Create windbreak vegetation barriers"
        ],
        "recommended_resources": [
            "Urban Air Quality Monitoring Handbook",
            "Phytoremediation Plant Species Guide",
            "Food Safety in Polluted Environments (FAO)"
        ]
    },
    "energy_costs": {
        "description": "High energy demands for artificial lighting and climate control",
        "solutions": ["solar_greenhouses", "passive_design", "LED_optimization"],
        "impact_score": 8,
        "detailed_strategies": [
            "Implement tiered lighting strategies",
            "Use thermal mass for temperature buffering",
            "Install photovoltaic-integrated growing systems",
            "Adopt circadian rhythm lighting schedules",
            "Utilize waste heat from adjacent buildings"
        ],
        "recommended_resources": [
            "Urban Ag Energy Efficiency Toolkit",
            "Passive Solar Greenhouse Design Guide",
            "LED Spectrum Optimization for Plant Growth"
        ]
    }
}

urban_best_practices = {
    "space_optimization": {
        "title": "Maximizing Growing Space in Urban Environments",
        "practices": [
            "Interplanting compatible crops to maximize space usage",
            "Vertical growing using trellises, walls, and stackable systems",
            "Succession planting to ensure continuous harvests",
            "Using space-efficient growing methods like Square Foot Gardening",
            "Selecting compact varieties bred for small spaces"
        ]
    },
    "water_conservation": {
        "title": "Urban Water Conservation Techniques",
        "practices": [
            "Collecting rainwater from rooftops and downspouts",
            "Using drip irrigation instead of overhead sprinklers",
            "Mulching to reduce evaporation and water needs",
            "Grouping plants with similar water requirements together",
            "Reusing household water when safe and appropriate"
        ]
    },
    "pest_management": {
        "title": "Organic Pest Management for Urban Agriculture",
        "practices": [
            "Companion planting to naturally deter pests",
            "Introducing beneficial insects like ladybugs and lacewings",
            "Using row covers to provide physical barriers",
            "Implementing regular crop rotation to break pest cycles",
            "Creating urban habitat for natural predators like birds"
        ]
    },
    "soil_health": {
        "title": "Building and Maintaining Healthy Urban Growing Medium",
        "practices": [
            "Composting food scraps and yard waste to create nutrient-rich soil",
            "Vermicomposting for apartment dwellers and small spaces",
            "Using cover crops to prevent erosion and add organic matter",
            "Applying organic amendments appropriate for specific crops",
            "Regular soil testing to monitor nutrient levels and contaminants"
        ]
    },
    "community_engagement": {
        "title": "Building Urban Agriculture Communities",
        "practices": [
            "Establishing seed-sharing networks among local growers",
            "Creating tool-lending libraries for shared equipment access",
            "Organizing knowledge-sharing workshops and skill exchanges",
            "Developing community composting programs",
            "Collaborating with local schools and community centers for education"
        ]
    },
    "energy_efficiency": {
        "title": "Optimizing Energy Use in Urban Farming Systems",
        "practices": [
            "Implement smart lighting systems with motion sensors",
            "Use waste heat from buildings for greenhouse heating",
            "Install renewable energy systems (solar/wind)",
            "Utilize thermal mass for natural temperature regulation",
            "Conduct regular energy audits of grow operations"
        ]
    },
    "waste_recycling": {
        "title": "Closed-Loop Resource Cycling in Urban Farms",
        "practices": [
            "Convert food waste to biochar for soil amendment",
            "Implement greywater recycling systems",
            "Repurpose spent growing media as mulch",
            "Use mushroom mycelium for substrate breakdown",
            "Create nutrient recovery systems from aquaponic waste"
        ]
    },
    "pollinator_support": {
        "title": "Urban Pollinator Habitat Creation",
        "practices": [
            "Install bee hotels for solitary pollinators",
            "Maintain flowering plants throughout seasons",
            "Create pesticide-free buffer zones",
            "Use nesting boxes for beneficial insects",
            "Implement nocturnal lighting that supports moths"
        ]
    },
    "tech_integration": {
        "title": "Smart Technology Implementation",
        "practices": [
            "Automated vertical farming climate control",
            "Sensor-based irrigation for green walls"
            "IoT sensor networks for real-time monitoring",
            "Automated nutrient dosing systems",
            "Machine learning-based pest detection",
            "Blockchain-enabled produce tracking",
            "Drone-assisted pollination in rooftop gardens"
        ]
    },
    "policy_navigation": {
        "title": "Navigating Urban Agricultural Regulations",
        "practices": [
            "Establish community land trusts",
            "Develop multi-use zoning proposals",
            "Create food production impact reports",
            "Leverage stormwater management incentives",
            "Partner with affordable housing initiatives"
        ]
    }
}