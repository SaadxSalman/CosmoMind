# Factory-Agent 🏭

A multi-agent system that autonomously manages a factory floor. Factory-Agent uses vision models to inspect products for defects, optimizes the assembly line in real-time, and handles logistics and supply chain management to maximize efficiency and reduce waste.

-----

## ✨ Features

  * **Autonomous Factory Management:** Manages the entire manufacturing process, from product inspection to supply chain optimization.
  * **Microscopic Defect Detection:** The **Inspection Agent** utilizes a **vision model** to identify product defects at a microscopic level, ensuring a high standard of quality control.
  * **Supply Chain & Logistics:** The **Logistics Agent** optimizes the supply chain and handles logistics, ensuring a smooth and efficient flow of materials.
  * **Real-time Assembly Optimization:** A **Robotics Control Agent** manages the assembly line, dynamically adjusting operations to maximize throughput.
  * **Data-Driven Decision-Making:** Leverages an **OLAP database** to analyze vast amounts of factory data and a local, fine-tuned **Gemma model** for intelligent decision-making.

-----

## ⚙️ Tech Stack

## 🛠️ Stack Expansion (Web Dashboard)

* **Framework:** [Next.js](https://nextjs.org/) (MERN Architecture)
* **Language:** [TypeScript](https://www.typescriptlang.org/)
* **Styling:** [Tailwind CSS](https://tailwindcss.com/)
* **Database:** MongoDB (part of MERN for persistent state management)
* **Runtime:** Node.js
* **Frontend:** [SvelteKit](https://kit.svelte.dev/)
* **Backend:** Rust
* **Data Analysis:** OLAP database
* **Core Model:** Fine-tuned Gemma model
* **Vision:** A vision model for inspection

-----

## 🚀 Getting Started

### Prerequisites

  * Rust
  * Node.js (for SvelteKit)
  * Access to a compatible OLAP database

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saadsalmanakram/Factory-Agent.git
    cd Factory-Agent
    ```
2.  **Set up the backend:**
    ```bash
    cargo build --release
    ```
3.  **Set up the frontend:**
    ```bash
    cd frontend
    npm install
    ```
4.  **Configure the OLAP database:**
    Follow the instructions in the documentation to connect the agent to your OLAP database.

### Configuration

Create a `.env` file to store your API keys and configuration variables for the Gemma model and other services.

### Usage

Run the Rust backend and the SvelteKit frontend to begin autonomous factory management and optimization.

-----

