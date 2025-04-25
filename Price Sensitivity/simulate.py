import tkinter as tk
from tkinter import messagebox, ttk
import random
import time
import csv
import os
from datetime import datetime
import numpy as np

class PriceSensitivityExperiment:
    def __init__(self, root):
        self.root = root
        self.root.title("Consumer Value Perception Study")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Experiment state
        self.participant_id = None
        self.current_trial = 0
        self.max_trials = 40  # Increased number of trials
        self.response_times = []
        self.choices = []
        self.shown_prices = []
        self.conditions = []
        self.product_features = []
        self.hypotheses = []
        
        # Test scenarios for Indian market with realistic pricing
        # Format: (price1, features1, price2, features2, hypothesis)
        self.test_scenarios = [
            # Core psychological barrier tests (₹999 vs ₹1000)
            (999, "Standard model", 1000, "Premium model", "psychological_barrier"),
            (999, "Special edition", 1000, "Limited edition", "psychological_barrier"),
            (4999, "Regular version", 5000, "Deluxe version", "psychological_barrier"),
            (9999, "Classic variant", 10000, "Enhanced variant", "psychological_barrier"),
            
            # Left-digit effect tests (various price points)
            (199, "Basic features", 200, "Basic features + 1 month warranty", "left_digit"),
            (299, "Standard package", 300, "Standard package + free delivery", "left_digit"),
            (499, "8GB storage", 500, "8GB storage + screen protector", "left_digit"),
            (1499, "Entry model", 1500, "Entry model + gift voucher", "left_digit"),
            
            # Charm pricing vs round pricing
            (499, "Core features", 500, "Core features + small accessory", "charm_vs_round"),
            (1999, "Standard variant", 2000, "Standard variant + extended warranty", "charm_vs_round"),
            (2999, "Base version", 3000, "Base version + priority service", "charm_vs_round"),
            
            # Price ending in 9 vs other endings
            (499, "Basic model", 495, "Basic model", "price_ending"),
            (999, "Standard edition", 990, "Standard edition", "price_ending"),
            (1499, "Regular package", 1490, "Regular package", "price_ending"),
            
            # Price alignment with quality perception
            (1499, "Entry level", 4999, "Premium level", "price_quality"),
            (999, "Budget friendly", 2999, "Professional grade", "price_quality"),
            (499, "Student edition", 1499, "Business edition", "price_quality"),
            
            # Discount framing
            (8999, "Original price: ₹10999 (₹2000 off)", 8999, "18% discount from ₹10999", "discount_framing"),
            (1499, "Sale price: ₹1499 (₹500 off)", 1499, "25% discount from ₹1999", "discount_framing"),
            (4999, "Special offer: ₹4999 (₹1000 off)", 4999, "17% discount from ₹5999", "discount_framing"),
            
            # Installment effect
            (5999, "One-time payment: ₹5999", 5999, "3 EMIs of ₹2000 each", "installment_effect"),
            (12999, "Full price: ₹12999", 12999, "6 EMIs of ₹2167 each", "installment_effect"),
            (24999, "Single payment: ₹24999", 24999, "12 EMIs of ₹2083 each", "installment_effect"),
            
            # Bundle pricing
            (2499, "Phone case", 2999, "Phone case + screen guard + earphones", "bundle_pricing"),
            (5999, "Main product", 6999, "Main product + 3 accessories", "bundle_pricing"),
            
            # Prestige pricing
            (999, "Standard model", 2999, "Premium model (same features)", "prestige_pricing"),
            (4999, "Regular edition", 9999, "Gold edition (identical specs)", "prestige_pricing"),
            
            # Price-sensitive segments
            (199, "Value option", 249, "Enhanced option", "price_sensitivity"),
            (99, "Budget choice", 149, "Quality choice", "price_sensitivity"),
            
            # Regional preference testing
            (499, "National brand", 449, "Local brand", "regional_preference"),
            (1999, "International model", 1799, "Indian brand model", "regional_preference"),
            
            # Free gift perception
            (1999, "Standard package", 2199, "Standard package + free gift worth ₹500", "free_gift"),
            (2999, "Base product", 3199, "Base product + complimentary item worth ₹1000", "free_gift"),
            
            # GST inclusion/exclusion framing
            (1499, "Price inclusive of GST", 1270, "Price: ₹1270 + 18% GST", "tax_framing"),
            (5999, "All taxes included", 5084, "Price: ₹5084 + 18% GST", "tax_framing")
        ]
        
        # Product categories for Indian market
        self.product_categories = [
            "Smartphone", "Bluetooth Earbuds", "Smart TV", 
            "Mixer Grinder", "Pressure Cooker", "Water Purifier", 
            "Air Conditioner", "Refrigerator", "Washing Machine",
            "Laptop", "Power Bank", "Induction Cooktop",
            "Digital Watch", "Air Purifier", "Ceiling Fan",
            "Rice Cooker", "Microwave Oven", "Electric Kettle",
            "Bluetooth Speaker", "Wi-Fi Router", "Fitness Band",
            "Gas Stove", "Geyser", "Sandwich Maker",
            "Hair Dryer", "Food Processor"
        ]
        
        # Welcome screen
        self.setup_welcome_screen()
        
    def setup_welcome_screen(self):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Header
        header = tk.Label(self.root, text="Consumer Value Perception Study", 
                         font=("Arial", 24, "bold"), bg="#f0f0f0", pady=20)
        header.pack()
        
        # Description (intentionally vague about true purpose)
        description = tk.Label(self.root, text="This research study examines how Indian consumers evaluate product options.\n"
                              "You will be asked to make a series of product choices based on your preferences.",
                         font=("Arial", 14), bg="#f0f0f0", pady=10, wraplength=700)
        description.pack()
        
        # Deception statement (ethical requirement but vague)
        deception_notice = tk.Label(self.root, text="Note: Some aspects of this study may not be fully disclosed until completion\n"
                                  "to avoid influencing your natural responses.",
                             font=("Arial", 12, "italic"), bg="#f0f0f0", fg="#555555", pady=5, wraplength=700)
        deception_notice.pack()
        
        # ID Frame
        id_frame = tk.Frame(self.root, bg="#f0f0f0", pady=20)
        id_frame.pack()
        
        id_label = tk.Label(id_frame, text="Participant ID:", font=("Arial", 12), bg="#f0f0f0")
        id_label.grid(row=0, column=0, padx=10)
        
        self.id_entry = tk.Entry(id_frame, font=("Arial", 12), width=10)
        self.id_entry.grid(row=0, column=1, padx=10)
        
        # Demographics (expanded for Indian context)
        demo_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        demo_frame.pack()
        
        tk.Label(demo_frame, text="Age range:", font=("Arial", 12), bg="#f0f0f0").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.age_var = tk.StringVar(value="25-34")
        age_options = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        age_dropdown = ttk.Combobox(demo_frame, textvariable=self.age_var, values=age_options, width=15)
        age_dropdown.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        tk.Label(demo_frame, text="Monthly income (₹):", font=("Arial", 12), bg="#f0f0f0").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.income_var = tk.StringVar(value="30,000-50,000")
        income_options = ["Below 15,000", "15,000-30,000", "30,000-50,000", "50,000-100,000", "Above 100,000"]
        income_dropdown = ttk.Combobox(demo_frame, textvariable=self.income_var, values=income_options, width=15)
        income_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        tk.Label(demo_frame, text="Location type:", font=("Arial", 12), bg="#f0f0f0").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.location_var = tk.StringVar(value="Urban")
        location_options = ["Metro city", "Urban", "Semi-urban", "Rural"]
        location_dropdown = ttk.Combobox(demo_frame, textvariable=self.location_var, values=location_options, width=15)
        location_dropdown.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        tk.Label(demo_frame, text="Online shopping frequency:", font=("Arial", 12), bg="#f0f0f0").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.shop_var = tk.StringVar(value="Monthly")
        shop_options = ["Weekly", "Monthly", "Quarterly", "Yearly", "Rarely"]
        shop_dropdown = ttk.Combobox(demo_frame, textvariable=self.shop_var, values=shop_options, width=15)
        shop_dropdown.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        # Start button
        start_button = tk.Button(self.root, text="Start Study", 
                               font=("Arial", 14, "bold"), 
                               command=self.start_experiment,
                               bg="#4CAF50", fg="white", padx=20, pady=10)
        start_button.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(self.root, text="Instructions:\n\n"
                              "1. You will be shown a series of products with two different options.\n"
                              "2. For each pair, select the option you would be most likely to purchase.\n"
                              "3. Consider both the price and described features for each option.\n"
                              "4. Make selections that reflect your actual purchasing preferences.\n"
                              "5. There are no right or wrong answers - we're interested in your honest opinions.",
                           font=("Arial", 12), bg="#f0f0f0", justify="left", wraplength=700)
        instructions.pack(pady=20)
        
    def start_experiment(self):
        # Validate participant ID
        if not self.id_entry.get().strip():
            messagebox.showerror("Error", "Please enter a participant ID")
            return
            
        self.participant_id = self.id_entry.get().strip()
        self.age_group = self.age_var.get()
        self.income_group = self.income_var.get()
        self.location_type = self.location_var.get()
        self.shopping_freq = self.shop_var.get()
        
        self.current_trial = 0
        self.response_times = []
        self.choices = []
        self.shown_prices = []
        self.conditions = []
        self.product_features = []
        self.hypotheses = []
        
        # Create results directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")
            
        # Run first trial
        self.run_trial()
        
    def create_color_block(self, color, size=100):
        """Create a colored block to represent product variations"""
        block = tk.Canvas(width=size, height=size, bg=color, highlightthickness=0)
        return block
        
    def run_trial(self):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Check if experiment is complete
        if self.current_trial >= self.max_trials:
            self.show_completion_screen()
            return
            
        # Select a random test scenario
        scenario = random.choice(self.test_scenarios)
        price1, features1, price2, features2, hypothesis = scenario
        
        # Select a random product category
        product_category = random.choice(self.product_categories)
        
        # Randomize left/right presentation
        if random.choice([True, False]):
            left_price, left_features = price1, features1
            right_price, right_features = price2, features2
            condition = "normal"
        else:
            left_price, left_features = price2, features2
            right_price, right_features = price1, features1
            condition = "reversed"
            
        # Generate random "model numbers" to make product comparison less obvious
        model_number1 = f"Model {chr(65 + random.randint(0, 25))}{random.randint(100, 999)}"
        model_number2 = f"Model {chr(65 + random.randint(0, 25))}{random.randint(100, 999)}"
        
        if left_price == right_price:
            condition = "equal_price"
        
        # Header
        progress_text = f"Decision {self.current_trial + 1} of {self.max_trials}"
        header = tk.Label(self.root, text=progress_text, font=("Arial", 14), bg="#f0f0f0", pady=10)
        header.pack()
        
        # Product category display
        category_label = tk.Label(self.root, text=product_category, font=("Arial", 18, "bold"), bg="#f0f0f0", pady=5)
        category_label.pack()
        
        # Instructions for this trial
        instructions = tk.Label(self.root, 
                              text="Which of these options would you be more likely to purchase?",
                              font=("Arial", 14), bg="#f0f0f0", pady=10)
        instructions.pack()
        
        # Container for options
        options_frame = tk.Frame(self.root, bg="#f0f0f0")
        options_frame.pack(pady=20, expand=True, fill="both")
        
        # Left option
        left_frame = tk.Frame(options_frame, bg="#f0f0f0", bd=2, relief="groove", padx=15, pady=15)
        left_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        
        # Left product color block - varies by trial for visual difference
        color1 = "#{:02x}{:02x}{:02x}".format(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        left_color = self.create_color_block(color1)
        left_color.pack(pady=10)
        
        # Left product details
        left_model = tk.Label(left_frame, text=model_number1, font=("Arial", 14, "bold"), bg="#f0f0f0")
        left_model.pack(pady=5)
        
        left_features_label = tk.Label(left_frame, text=left_features, font=("Arial", 12), bg="#f0f0f0", wraplength=250)
        left_features_label.pack(pady=10)
        
        left_price_label = tk.Label(left_frame, text=f"₹{left_price}", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#d35400")
        left_price_label.pack(pady=10)
        
        left_button = tk.Button(left_frame, text="Select This Option", 
                             font=("Arial", 12), 
                             bg="#3498db", fg="white", padx=10, pady=5,
                             command=lambda: self.record_choice("left", left_price, right_price, left_features, right_features, condition, hypothesis))
        left_button.pack(pady=15)
        
        # Right option
        right_frame = tk.Frame(options_frame, bg="#f0f0f0", bd=2, relief="groove", padx=15, pady=15)
        right_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        
        # Right product color block - slightly different color
        color2 = "#{:02x}{:02x}{:02x}".format(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        right_color = self.create_color_block(color2)
        right_color.pack(pady=10)
        
        # Right product details
        right_model = tk.Label(right_frame, text=model_number2, font=("Arial", 14, "bold"), bg="#f0f0f0")
        right_model.pack(pady=5)
        
        right_features_label = tk.Label(right_frame, text=right_features, font=("Arial", 12), bg="#f0f0f0", wraplength=250)
        right_features_label.pack(pady=10)
        
        right_price_label = tk.Label(right_frame, text=f"₹{right_price}", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#d35400")
        right_price_label.pack(pady=10)
        
        right_button = tk.Button(right_frame, text="Select This Option", 
                              font=("Arial", 12), 
                              bg="#3498db", fg="white", padx=10, pady=5,
                              command=lambda: self.record_choice("right", left_price, right_price, left_features, right_features, condition, hypothesis))
        right_button.pack(pady=15)
        
        # Make sure columns have equal width
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        
        # Deception - add fake ratings to make choices less obvious
        if random.choice([True, False]):
            tk.Label(left_frame, text="★★★★☆", font=("Arial", 14), bg="#f0f0f0", fg="gold").pack()
            tk.Label(right_frame, text="★★★★★", font=("Arial", 14), bg="#f0f0f0", fg="gold").pack()
        else:
            tk.Label(left_frame, text="★★★★★", font=("Arial", 14), bg="#f0f0f0", fg="gold").pack()
            tk.Label(right_frame, text="★★★★☆", font=("Arial", 14), bg="#f0f0f0", fg="gold").pack()
        
        # Add Indian-specific shopping elements
        review_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        review_frame.pack(fill="x")
        
        reviews = [
            "Certified BIS approved product",
            "Free delivery on orders above ₹499",
            "Exchange offer available",
            "Bank offers: 10% off with HDFC cards",
            "No Cost EMI available",
            "Special offer ends in 2 days",
            "4.5+ rating on Flipkart/Amazon",
            "Genuine product with manufacturer warranty",
            "Made in India product",
            "Energy efficient - 5 star rating"
        ]
        review_text = random.choice(reviews)
        review_label = tk.Label(review_frame, text=review_text, font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#555555")
        review_label.pack()
        
        # Store trial start time
        self.trial_start_time = time.time()
        
    def record_choice(self, position, left_price, right_price, left_features, right_features, condition, hypothesis):
        # Calculate response time
        response_time = time.time() - self.trial_start_time
        
        # Record data based on position selected
        if position == "left":
            chosen_price = left_price
            chosen_features = left_features
            other_price = right_price
            other_features = right_features
        else:
            chosen_price = right_price
            chosen_features = right_features
            other_price = left_price
            other_features = left_features
        
        # Record full data
        price_pair = (left_price, right_price)
        features_pair = (left_features, right_features)
        
        self.choices.append((position, chosen_price, chosen_features))
        self.response_times.append(response_time)
        self.shown_prices.append(price_pair)
        self.conditions.append(condition)
        self.product_features.append(features_pair)
        self.hypotheses.append(hypothesis)
        
        # Increment trial counter
        self.current_trial += 1
        
        # Occasionally insert a satisfaction question to disguise true purpose
        if self.current_trial % 8 == 0 and self.current_trial < self.max_trials:
            self.show_satisfaction_question()
        # Every 15 trials, show a preference question
        elif self.current_trial % 15 == 0 and self.current_trial < self.max_trials:
            self.show_preference_question()
        else:
            # Short delay before next trial
            self.root.after(500, self.run_trial)
            
    def show_satisfaction_question(self):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Satisfaction question (distraction from price focus)
        header = tk.Label(self.root, text="Quick Feedback", font=("Arial", 18, "bold"), bg="#f0f0f0", pady=20)
        header.pack()
        
        question = tk.Label(self.root, 
                          text="How satisfied are you with the product options you've seen so far?",
                          font=("Arial", 14), bg="#f0f0f0", pady=10, wraplength=600)
        question.pack()
        
        # Rating scale
        scale_frame = tk.Frame(self.root, bg="#f0f0f0", pady=20)
        scale_frame.pack()
        
        self.satisfaction_var = tk.IntVar(value=3)
        satisfaction_scale = tk.Scale(scale_frame, from_=1, to=5, orient="horizontal", 
                                   length=400, tickinterval=1, resolution=1, 
                                   variable=self.satisfaction_var,
                                   label="Satisfaction Level",
                                   font=("Arial", 12), bg="#f0f0f0")
        satisfaction_scale.pack()
        
        # Labels for scale
        labels_frame = tk.Frame(scale_frame, bg="#f0f0f0")
        labels_frame.pack(fill="x", pady=5)
        
        tk.Label(labels_frame, text="Very Dissatisfied", font=("Arial", 10), bg="#f0f0f0").pack(side="left")
        tk.Label(labels_frame, text="Very Satisfied", font=("Arial", 10), bg="#f0f0f0").pack(side="right")
        
        # Continue button
        continue_button = tk.Button(self.root, text="Continue", 
                                  font=("Arial", 14), 
                                  bg="#4CAF50", fg="white", padx=20, pady=10,
                                  command=self.run_trial)
        continue_button.pack(pady=30)
        
    def show_preference_question(self):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Preference question (to gather more insights on Indian market)
        header = tk.Label(self.root, text="Shopping Preferences", font=("Arial", 18, "bold"), bg="#f0f0f0", pady=20)
        header.pack()
        
        questions = [
            "When purchasing electronics, which factor matters most to you?",
            "How do you typically prefer to pay for purchases above ₹10,000?",
            "Which shopping platform do you prefer for most purchases?",
            "How important is a product being 'Made in India' to your purchase decision?"
        ]
        
        options_sets = [
            ["Price", "Brand", "Features", "Warranty", "Reviews"],
            ["Full payment", "EMI", "Buy Now Pay Later", "Credit Card", "Cash on Delivery"],
            ["Amazon", "Flipkart", "Local retail store", "Brand website", "Other online marketplaces"],
            ["Very important", "Somewhat important", "Neutral", "Not very important", "Not important at all"]
        ]
        
        # Randomly choose a question
        question_index = random.randint(0, len(questions)-1)
        question_text = questions[question_index]
        options = options_sets[question_index]
        
        question_label = tk.Label(self.root, 
                                text=question_text,
                                font=("Arial", 14), bg="#f0f0f0", pady=10, wraplength=600)
        question_label.pack(pady=20)
        
        # Create radio buttons for options
        option_frame = tk.Frame(self.root, bg="#f0f0f0")
        option_frame.pack(pady=10)
        
        self.preference_var = tk.StringVar(value=options[0])
        
        for option in options:
            radio = tk.Radiobutton(option_frame, text=option, variable=self.preference_var, 
                                 value=option, font=("Arial", 12), bg="#f0f0f0",
                                 padx=10, pady=5)
            radio.pack(anchor="w")
        
        # Continue button
        continue_button = tk.Button(self.root, text="Continue", 
                                  font=("Arial", 14), 
                                  bg="#4CAF50", fg="white", padx=20, pady=10,
                                  command=self.run_trial)
        continue_button.pack(pady=30)
        
    def show_completion_screen(self):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Save data
        self.save_data()
            
        # Completion message
        completion = tk.Label(self.root, text="Study Complete!", 
                            font=("Arial", 24, "bold"), bg="#f0f0f0", pady=20)
        completion.pack()
        
        # Now reveal the true purpose of the study
        reveal = tk.Label(self.root, text="Study Purpose Disclosure", 
                        font=("Arial", 16, "bold"), bg="#f0f0f0", pady=15)
        reveal.pack()
        
        explanation = tk.Label(self.root, 
                             text="This study was designed to investigate multiple psychological pricing hypotheses in the Indian consumer market.\n\n"
                             "We tested various aspects including:\n"
                             "• Psychological pricing thresholds (e.g., ₹999 vs ₹1000)\n"
                             "• Left-digit effect across different price levels\n"
                             "• Price presentation formats (EMIs, discounts, tax-inclusive vs exclusive)\n"
                             "• Value perception of bundled offers\n"
                             "• Indian market-specific preferences (local vs international brands)\n\n"
                             "Your participation helps understand consumer psychology in the Indian retail context.",
                             font=("Arial", 12), bg="#f0f0f0", wraplength=700, justify="left")
        explanation.pack(pady=10)
        
        # Thank you
        thank_you = tk.Label(self.root, text="Thank you for your valuable contribution to this research.", 
                           font=("Arial", 14), bg="#f0f0f0", pady=10)
        thank_you.pack()
        
        # Exit button
        exit_button = tk.Button(self.root, text="Exit Study", 
                              font=("Arial", 14, "bold"), 
                              command=self.root.destroy,
                              bg="#e74c3c", fg="white", padx=20, pady=10)
        exit_button.pack(pady=20)
        
    def save_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/participant_{self.participant_id}_{timestamp}.csv"
        
        # Create summary analysis
        hypotheses_results = {}
        for i, hypothesis in enumerate(self.hypotheses):
            if hypothesis not in hypotheses_results:
                hypotheses_results[hypothesis] = {
                    'total': 0,
                    'lower_price_chosen': 0,
                    'higher_price_chosen': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            hypotheses_results[hypothesis]['total'] += 1
            
            # Get price data
            left_price, right_price = self.shown_prices[i]
            position, chosen_price, _ = self.choices[i]
            
            # Record which price was chosen
            if chosen_price == min(left_price, right_price):
                hypotheses_results[hypothesis]['lower_price_chosen'] += 1
            else:
                hypotheses_results[hypothesis]['higher_price_chosen'] += 1
                
            # Record response time
            hypotheses_results[hypothesis]['response_times'].append(self.response_times[i])
        
        # Calculate averages
        for hypothesis in hypotheses_results:
            if hypotheses_results[hypothesis]['total'] > 0:
                hypotheses_results[hypothesis]['avg_response_time'] = sum(hypotheses_results[hypothesis]['response_times']) / hypotheses_results[hypothesis]['total']
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header row with demographic info
            writer.writerow(['Participant ID', 'Age Group', 'Income Group', 'Location Type', 'Shopping Frequency'])
            writer.writerow([self.participant_id, self.age_group, self.income_group, self.location_type, self.shopping_freq])
            writer.writerow([]) # Blank row
            
            # Summary results by hypothesis
            writer.writerow(['HYPOTHESIS SUMMARY'])
if __name__ == '__main__':
    root = tk.Tk()
    app = PriceSensitivityExperiment(root)
    root.mainloop()