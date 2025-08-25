# AI-Powered Urban Agriculture Assistant with RAG

[[Project Description and Overview]](#project-description-and-overview) | 
[[Tech Stack]](#tech-stack) | 
[[Workflow]](#workflow) |
[[Example Conversation]](#example-conversation)

<a id="project-description-and-overview"></a>
## 1) Project Description and Overview

This project implements a Retrieval-Augmented Generation (RAG) system for urban agriculture expertise, combining vector database retrieval with AI reasoning. The system provides context-aware solutions through:

1. **Multi-Collection Retrieval**: Queries 4 specialized knowledge bases:
   - Urban agriculture methods
   - Suitable crops
   - Common challenges
   - Best practices

2. **RAG Agent**: Synthesizes retrieved information using quantitative metrics and practical considerations

3. **Contextual Analysis**: Maintains awareness of document relevance scores and data types to weight information appropriately

The system handles complex urban farming questions through semantic search and evidence-based recommendations.

<a id="tech-stack"></a>
## 2) Tech Stack

- **Core Framework**: 
  - Autogen (Agent Orchestration)
  - Qdrant (Vector Database)
  - Python 3.11+

- **AI Services**:
  - Azure OpenAI (GPT-4o)
  - Sentence Transformers (Embeddings)

- **Key Components**:
  - `qdrant_client`: Vector search operations
  - `sentence-transformers`: Text embedding generation
  - `autogen`: Agent infrastructure

- **Knowledge Bases**:
  - 4 Qdrant collections with urban agriculture data:
    - urban_ag_methods
    - urban_crops
    - urban_challenges  
    - urban_best_practices

<a id="workflow"></a>
## 3) Workflow

1. **Query Processing**:
   - User question is embedded using all-MiniLM-L6-v2 model
   - Parallel searches across 4 knowledge bases

2. **Context Retrieval**:
   - Top 3 results per collection retrieved
   - Results sorted by relevance score
   - Metadata-aware aggregation

3. **Response Generation**:
   - Agent receives structured context with:
     - Source collection type
     - Relevance scores
     - Text content
   - Generates comparative analysis considering:
     - Space efficiency metrics
     - Cost considerations
     - Implementation challenges

<a id="example-conversation"></a>
## 4)  Example Conversation

**User Query**:  
"What's the most space-efficient method for growing strawberries in a polluted urban area?"

**Agent Response**
```plaintext
Considering the constraints of polluted urban areas and the need for maximizing space efficiency, below are the most viable methods for growing strawberries along with their analysis:

1. **Aeroponic Systems**
   - **Space Efficiency**: Highest (0.97)
   - **Water Usage**: Lowest (0.03)
   - **Energy Requirement**: High (0.9)
   - **Advantages**:
     * Extremely space-efficient, making it ideal for tightly packed urban settings.
     * Significant water savings and faster growth rates compared to hydroponics.
     * Ability to maintain nutrient-rich environments that enhance plant health.
     * Reduced disease transmission risks.
   - **Disadvantages**:
     * High initial costs ($1200-$5000 per square meter).
     * Requires precise misting schedule and is vulnerable to pump failures.
     * Needs advanced technical knowledge to maintain.
   - **Ideal Locations**: High-tech research facilities, modular shipping container farms, underground growing bunkers.

2. **Vertical Farming**
   - **Space Efficiency**: High (0.90)
   - **Water Usage**: Low (0.1)
   - **Energy Requirement**: Moderate to high (0.8)
   - **Advantages**:
     * Efficiently utilizes limited urban space through vertical stacking.
     * Controlled environment for year-round production.
     * Protection from external pollutants, pests, and weather extremes.
     * Reduced water use through recirculation.
   - **Disadvantages**:
     * Initial setup costs ($500-$3000 per square meter).
     * Requires technical knowledge and energy for lighting and climate control.
     * Limited to certain crop types primarily leafy greens, herbs, and strawberries.
   - **Ideal Locations**: Warehouses, abandoned buildings, basement spaces.

3. **Vegetated Vertical Surfaces (Green Walls)**
   - **Space Efficiency**: High (0.92)
   - **Water Usage**: Moderate (0.4)
   - **Energy Requirement**: Low (0.3)
   - **Advantages**:
     * Enhances building insulation and urban air quality.
     * Adds aesthetic value and noise barrier functionality.
     * Supports biodiversity with pollinator-friendly plants.
   - **Disadvantages**:
     * Specialized irrigation system requirements.
     * Potential structural stress on building walls.
     * Maintenance access challenges.
     * Wind exposure issues at height.
   - **Ideal Locations**: Building exteriors, interior atriums, public transportation hubs, retail spaces.

### Summary
For the highest space efficiency, **Aeroponic Systems** are the most effective, although they come with the highest startup costs and technical requirements. They also offer water savings and fast growth rates, making them particularly suitable for urban areas where every square meter counts.
**Vertical Farming** is a less costly alternative with slightly lower space efficiency but provides controlled conditions and protection from pollutants. **Vegetated Vertical Surfaces** offer both environmental and aesthetic benefits, with moderate space efficiency and lower energy requirements.

**Aeroponics** is recommended for top-tier space efficiency if budget and technical know-how permit. However, **Vertical Farming** remains a solid choice for balance between cost, efficiency, and manageability in polluted urban setups.
```