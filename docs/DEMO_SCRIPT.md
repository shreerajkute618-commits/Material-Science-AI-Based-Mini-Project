# TWO-MINUTE COLLEGE PROJECT DEMONSTRATION SCRIPT
## AI-Based Identification and Sorting of Recyclable Materials
### Materials Science and Engineering Laboratory

---

### **Demonstration Setup (Prior to presentation)**
- Ensure the Streamlit web application is running in your browser: `http://localhost:8501`.
- Have sample images ready in a folder or have a material sample (plastic bottle, aluminum can, paper) handy for live webcam demonstration.

---

### **Minute 0:00 – 0:30 | Introduction & Motivation**
*(Stand with confidence, display the Dashboard page)*

> "Respected Evaluators and Professors, good morning/afternoon. 
> 
> Today, I am proud to present our Materials Science and Engineering mini-project: **'AI-Based Identification and Sorting of Recyclable Materials'**.
> 
> In modern Materials Recovery Facilities, manual waste picking is dangerous, unhygienic, and results in heavy cross-contamination of secondary raw materials. 
> 
> Our project solves this problem by combining Computer Vision, Deep Learning, and fundamental Materials Science principles to automatically identify, evaluate, and route recyclable waste."

---

### **Minute 0:30 – 1:00 | Core Technology & AI Classification Demonstration**
*(Navigate to '🔍 Material Identification' page)*

> "Let us test the system live. I will upload an image of a post-consumer material sample.
> 
> *(Upload an image or capture a sample with the webcam)*
> 
> In less than 100 milliseconds, our fine-tuned **MobileNetV2** deep convolutional neural network analyzes the sample's surface geometry, texture, and specular reflections. 
> 
> Notice that the system doesn't just give a blind label:
> 1. It gives us a **calibrated probability distribution** across all six circular economy categories.
> 2. It incorporates an **uncertainty safety guardrail**: if the confidence drops below 50%, it flags a warning rather than making an overconfident mistake."

---

### **Minute 1:00 – 1:30 | Materials Science Knowledge & Sorting Simulation**
*(Scroll down to the Materials Science cards, then switch to '⚙️ Automated Sorting Simulation')*

> "What truly distinguishes this system as a Materials Science and Engineering project is what happens next.
> 
> The application retrieves the material's **chemical composition and microstructure**—such as crystalline metallic lattices for aluminum or macromolecular hydrocarbon chains for plastics. It displays critical physical properties including density, thermal conductivity, and melting point, along with the precise industrial recycling protocol and environmental degradation lifespan.
> 
> Now, looking at our **Automated Sorting Digital Twin**:
> The AI detection immediately triggers the simulated conveyor chute diverter, routing the material to Chute #1 for optical NIR and sink-float separation."

---

### **Minute 1:30 – 2:00 | Model Performance, Metrics & Conclusion**
*(Navigate to '📊 Dataset & Model Performance' page)*

> "Finally, let's examine our scientific evaluation. 
> 
> The model was trained using Transfer Learning on the benchmark Stanford TrashNet dataset of 2,527 images, rigorously divided into training, validation, and a separate test set with **zero data leakage**. 
> 
> As you can see from our confusion matrix and performance curves, the model delivers high precision and recall across all major recyclable streams. Every classification is logged in our session history table and can be exported as a CSV report for facility auditing.
> 
> In conclusion, this project provides a scalable, sustainable, and data-driven approach to circular materials management. Thank you, and I am now ready for your questions."

---

### **Pro-Tips for High Marks:**
1. **Speak clearly and at a measured pace**: Don't rush; pause after showing key features.
2. **Emphasize Materials Science**: Mention terms like *FCC crystalline lattice*, *van der Waals bonding*, *macromolecular degradation*, and *cullet thermodynamics*.
3. **Show the Confusion Matrix**: Demonstrates that you performed rigorous empirical testing.
