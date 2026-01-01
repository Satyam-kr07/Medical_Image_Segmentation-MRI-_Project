import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import time

class MRIInterpolator:
    """Advanced MRI image interpolator with structure-preserving morphological interpolation"""
    
    def interpolate(self, img1_path, img2_path, alpha=0.5):
        """
        Generate a naturally interpolated image between two MRI images with structure preservation
        
        Parameters:
        -----------
        img1_path : str
            Path to the first image
        img2_path : str
            Path to the second image
        alpha : float
            Interpolation factor between 0 and 1
            
        Returns:
        --------
        numpy.ndarray
            The interpolated image with preserved structures
        """
        # Read images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if images were loaded properly
        if img1 is None:
            raise ValueError("Could not read first image")
        if img2 is None:
            raise ValueError("Could not read second image")
            
        # Ensure both images have the same dimensions
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to float for better precision in calculations
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Create intensity-normalized versions for better structure analysis
        img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        
        # Multi-scale decomposition for structure preservation
        # Level 1: Base structures
        img1_blur = cv2.GaussianBlur(img1_norm, (5, 5), 0)
        img2_blur = cv2.GaussianBlur(img2_norm, (5, 5), 0)
        img1_detail = img1_norm - img1_blur
        img2_detail = img2_norm - img2_blur
        
        # Level 2: Finer structures
        img1_blur2 = cv2.GaussianBlur(img1_norm, (15, 15), 0)
        img2_blur2 = cv2.GaussianBlur(img2_norm, (15, 15), 0)
        img1_mid = img1_blur - img1_blur2
        img2_mid = img2_blur - img2_blur2
        
        # Base interpolation (preserves average intensity)
        base_interp = (1 - alpha) * img1_blur2 + alpha * img2_blur2
        mid_interp = (1 - alpha) * img1_mid + alpha * img2_mid
        detail_interp = (1 - alpha) * img1_detail + alpha * img2_detail
        
        # Recombine multi-scale interpolation
        multi_scale_result = base_interp + mid_interp + detail_interp
        
        # Structural correlation map to identify corresponding regions
        # Use normalized cross-correlation to find matching structures
        window_size = 7
        correlation_map = np.zeros_like(img1)
        pad_size = window_size // 2
        padded1 = cv2.copyMakeBorder(img1_norm, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        padded2 = cv2.copyMakeBorder(img2_norm, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        
        # Sample correlation at grid points to save computation
        grid_step = 3
        for i in range(0, img1.shape[0], grid_step):
            for j in range(0, img1.shape[1], grid_step):
                patch1 = padded1[i:i+window_size, j:j+window_size]
                patch2 = padded2[i:i+window_size, j:j+window_size]
                if np.std(patch1) > 1 and np.std(patch2) > 1:  # Skip uniform regions
                    corr = cv2.matchTemplate(patch1, patch2, cv2.TM_CCORR_NORMED)[0][0]
                    correlation_map[i:i+grid_step, j:j+grid_step] = corr
        
        # Smooth and normalize correlation map
        correlation_map = cv2.GaussianBlur(correlation_map, (7, 7), 0)
        correlation_map = cv2.normalize(correlation_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Weight map for morphological interpolation based on image structures
        # Extract morphological features (edges, shapes)
        edges1 = cv2.Canny(img1_norm.astype(np.uint8), 50, 150)
        edges2 = cv2.Canny(img2_norm.astype(np.uint8), 50, 150)
        
        # Dilate edges to get regions around key structures
        kernel = np.ones((3, 3), np.uint8)
        edges1_dilated = cv2.dilate(edges1, kernel, iterations=2)
        edges2_dilated = cv2.dilate(edges2, kernel, iterations=2)
        
        # Combined structure map
        structure_map = (edges1_dilated.astype(np.float32) + edges2_dilated.astype(np.float32)) / 255.0
        structure_map = cv2.GaussianBlur(structure_map, (5, 5), 0)
        structure_map = cv2.normalize(structure_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate mean and std for both images to maintain correct intensity statistics
        mean1, std1 = np.mean(img1), np.std(img1)
        mean2, std2 = np.mean(img2), np.std(img2)
        target_mean = (1 - alpha) * mean1 + alpha * mean2
        target_std = (1 - alpha) * std1 + alpha * std2
        
        # Direct linear interpolation as baseline
        linear_interp = (1 - alpha) * img1 + alpha * img2
        
        # Adaptive blending of interpolation methods
        # Higher weight to multi-scale in structure areas, higher weight to linear in smooth areas
        blend_weight = 0.7 * structure_map + 0.3 * (1 - correlation_map)
        blend_weight = cv2.GaussianBlur(blend_weight, (7, 7), 0)
        
        # Final interpolation with adaptive blending
        result = blend_weight * multi_scale_result + (1 - blend_weight) * linear_interp
        
        # Match statistics to ensure correct intensity range
        result_mean = np.mean(result)
        result_std = np.std(result)
        if result_std > 0:
            result = ((result - result_mean) / result_std * target_std) + target_mean
        
        # Final cleanup and range adjustment
        result = np.clip(result, 0, 255)
        result_uint8 = result.astype(np.uint8)
        
        # Apply subtle edge-preserving filter for final image
        final_result = cv2.bilateralFilter(result_uint8, 5, 30, 30)
        
        return final_result

class MRIInterpolationApp:
   
    def __init__(self, root):
        self.root = root
        self.root.title("MRI Image Interpolation Tool")
        self.root.geometry("1200x800")  # Increased height to accommodate both sections

        # Initialize variables
        self.img1_path = None
        self.img2_path = None
        self.interpolated_img = None
        self.transition_frames = []  # Store transition frames
        self.current_frame_index = 0  # Current frame in transitions view

        # Initialize interpolator
        self.interpolator = MRIInterpolator()  # Add this line here

        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create the controls frame
        self.create_controls_frame()

        # Create the right side panel
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Divide right panel into top and bottom sections
        self.top_panel = ttk.LabelFrame(self.right_panel, text="Main Interpolation Result")
        self.top_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.bottom_panel = ttk.LabelFrame(self.right_panel, text="Transition Frames")
        self.bottom_panel.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Create the image displays
        self.create_main_image_display()
        self.create_transition_display()

        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please select two MRI images to begin.")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    
    def create_controls_frame(self):
        # Create the left side control panel
        self.controls_frame = ttk.Frame(self.main_frame, padding="10", relief="raised", width=300)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.controls_frame.pack_propagate(False)  # Prevent shrinking
        
        # Title
        ttk.Label(self.controls_frame, text="MRI Interpolation Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Image selection section
        ttk.Label(self.controls_frame, text="Step 1: Select Reference Images", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        # First image selection
        img1_frame = ttk.Frame(self.controls_frame)
        img1_frame.pack(fill=tk.X, pady=5)
        ttk.Label(img1_frame, text="Reference Image 1:").pack(side=tk.LEFT)
        ttk.Button(img1_frame, text="Browse...", command=self.load_image1).pack(side=tk.RIGHT)
        
        self.img1_label = ttk.Label(self.controls_frame, text="No image selected")
        self.img1_label.pack(fill=tk.X, pady=2)
        
        # Second image selection
        img2_frame = ttk.Frame(self.controls_frame)
        img2_frame.pack(fill=tk.X, pady=5)
        ttk.Label(img2_frame, text="Reference Image 2:").pack(side=tk.LEFT)
        ttk.Button(img2_frame, text="Browse...", command=self.load_image2).pack(side=tk.RIGHT)
        
        self.img2_label = ttk.Label(self.controls_frame, text="No image selected")
        self.img2_label.pack(fill=tk.X, pady=2)
        
        # Interpolation parameters section
        ttk.Label(self.controls_frame, text="Step 2: Configure Parameters", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(20, 5))
        
        # Alpha parameter (interpolation factor)
        alpha_frame = ttk.Frame(self.controls_frame)
        alpha_frame.pack(fill=tk.X, pady=5)
        ttk.Label(alpha_frame, text="Interpolation Factor:").pack(side=tk.LEFT)
        
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_scale = ttk.Scale(alpha_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                               variable=self.alpha_var, command=self.update_alpha_label)
        alpha_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.alpha_label = ttk.Label(alpha_frame, text="0.5")
        self.alpha_label.pack(side=tk.RIGHT, padx=5)
        
        # Number of transition frames
        frames_frame = ttk.Frame(self.controls_frame)
        frames_frame.pack(fill=tk.X, pady=5)
        ttk.Label(frames_frame, text="Transition Frames:").pack(side=tk.LEFT)
        
        self.frames_var = tk.IntVar(value=5)
        frames_scale = ttk.Scale(frames_frame, from_=2, to=8, orient=tk.HORIZONTAL,
                                variable=self.frames_var, command=self.update_frames_label)
        frames_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.frames_label = ttk.Label(frames_frame, text="5")
        self.frames_label.pack(side=tk.RIGHT, padx=5)
        
        # Process buttons
        ttk.Label(self.controls_frame, text="Step 3: Generate Results", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(20, 5))
        
        button_frame = ttk.Frame(self.controls_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.process_button = ttk.Button(button_frame, text="Generate Interpolated Image", 
                                        command=self.process_images, state=tk.DISABLED)
        self.process_button.pack(fill=tk.X, pady=5)
        
        self.generate_transitions_button = ttk.Button(button_frame, text="Generate Transition Frames", 
                                                    command=self.generate_transition_frames, state=tk.DISABLED)
        self.generate_transitions_button.pack(fill=tk.X, pady=5)
        
        # Save button
        self.save_button = ttk.Button(self.controls_frame, text="Save Current Result", 
                                     command=self.save_result, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)
        
        # Method explanation
        ttk.Label(self.controls_frame, text="Information:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(20, 5))
        
        info_text = """This tool uses adaptive interpolation between two MRI images.

The algorithm preserves important structural features while blending the images smoothly.

The main display shows your source images and the interpolated result.

The transition frames display shows a sequence of intermediate states between the two source images.

Use the transition frames slider to control how many intermediate frames to show between the two images.

You can navigate through frames with the arrows and save any frame."""
        
        self.method_info = tk.Text(self.controls_frame, height=10, width=30, wrap=tk.WORD)
        self.method_info.pack(fill=tk.BOTH, expand=True, pady=5)
        self.method_info.insert(tk.END, info_text)
        self.method_info.config(state=tk.DISABLED)
    
    def create_main_image_display(self):
        # Create the top image display for main interpolation result
        self.main_fig = plt.Figure(figsize=(10, 4), dpi=100)
        self.main_canvas = FigureCanvasTkAgg(self.main_fig, master=self.top_panel)
        self.main_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.ax1 = self.main_fig.add_subplot(131)
        self.ax2 = self.main_fig.add_subplot(132)
        self.ax3 = self.main_fig.add_subplot(133)
        
        self.ax1.set_title("Reference Image 1")
        self.ax2.set_title("Interpolated Result")
        self.ax3.set_title("Reference Image 2")
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.axis('off')
        
        self.main_fig.tight_layout()
        self.main_canvas.draw()
    
    def create_transition_display(self):
        # Create the bottom frame for transition frames display
        self.transition_display_frame = ttk.Frame(self.bottom_panel)
        self.transition_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create navigation controls
        self.nav_frame = ttk.Frame(self.transition_display_frame)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.prev_button = ttk.Button(self.nav_frame, text="◀ Previous", command=self.prev_transition)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.transition_indicator = ttk.Label(self.nav_frame, text="No transitions generated")
        self.transition_indicator.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        self.next_button = ttk.Button(self.nav_frame, text="Next ▶", command=self.next_transition)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        
        # Create the transitions figure
        self.transitions_fig = plt.Figure(figsize=(10, 3), dpi=100)
        self.transitions_canvas = FigureCanvasTkAgg(self.transitions_fig, master=self.transition_display_frame)
        self.transitions_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initially no frames to show
        self.update_transition_display()
    
    def update_transition_display(self):
        # Clear the figure
        self.transitions_fig.clear()
        
        if not self.transition_frames:
            # No frames to display yet
            ax = self.transitions_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No transition frames generated yet.\nUse 'Generate Transition Frames' button after processing.", 
                   horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.axis('off')
        else:
            # Determine how many frames to show in a row (usually 3-5)
            display_frames = min(5, len(self.transition_frames))
            
            # Calculate which frames to show based on current_frame_index
            total_frames = len(self.transition_frames)
            
            # Ensure we don't go out of bounds
            start_idx = max(0, min(self.current_frame_index, total_frames - display_frames))
            end_idx = min(start_idx + display_frames, total_frames)
            
            # Create one subplot for each frame to display
            for i, frame_idx in enumerate(range(start_idx, end_idx)):
                ax = self.transitions_fig.add_subplot(1, display_frames, i+1)
                ax.imshow(self.transition_frames[frame_idx], cmap='gray')
                
                # Calculate alpha value for this frame
                if total_frames > 1:
                    alpha = frame_idx / (total_frames - 1)
                    ax.set_title(f"α={alpha:.2f}")
                
                ax.axis('off')
            
            # Update the indicator
            self.transition_indicator.config(
                text=f"Showing frames {start_idx+1}-{end_idx} of {total_frames}"
            )
        
        self.transitions_fig.tight_layout()
        self.transitions_canvas.draw()
    
    def prev_transition(self):
        # Show previous set of transition frames
        if not self.transition_frames:
            return
        
        # Move back by the number of frames we display at once (usually 5)
        display_count = min(5, len(self.transition_frames))
        self.current_frame_index = max(0, self.current_frame_index - display_count)
        self.update_transition_display()
    
    def next_transition(self):
        # Show next set of transition frames
        if not self.transition_frames:
            return
        
        # Move forward by the number of frames we display at once (usually 5)
        display_count = min(5, len(self.transition_frames))
        max_start_idx = max(0, len(self.transition_frames) - display_count)
        self.current_frame_index = min(max_start_idx, self.current_frame_index + display_count)
        self.update_transition_display()
    
    def load_image1(self):
        # Open file dialog to select first image
        filepath = filedialog.askopenfilename(
            title="Select First MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if filepath:
            self.img1_path = filepath
            filename = os.path.basename(filepath)
            self.img1_label.config(text=f"Selected: {filename}")
            self.status_var.set(f"Loaded first reference image: {filename}")
            self.update_image_display()
            self.check_process_button()
    
    def load_image2(self):
        # Open file dialog to select second image
        filepath = filedialog.askopenfilename(
            title="Select Second MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if filepath:
            self.img2_path = filepath
            filename = os.path.basename(filepath)
            self.img2_label.config(text=f"Selected: {filename}")
            self.status_var.set(f"Loaded second reference image: {filename}")
            self.update_image_display()
            self.check_process_button()
    
    def update_image_display(self):
        # Clear the axes
        self.ax1.clear()
        self.ax3.clear()
        self.ax1.axis('off')
        self.ax3.axis('off')
        
        # Display the loaded images
        if self.img1_path:
            img1 = cv2.imread(self.img1_path, cv2.IMREAD_GRAYSCALE)
            self.ax1.imshow(img1, cmap='gray')
            self.ax1.set_title("Reference Image 1")
        
        if self.img2_path:
            img2 = cv2.imread(self.img2_path, cv2.IMREAD_GRAYSCALE)
            self.ax3.imshow(img2, cmap='gray')
            self.ax3.set_title("Reference Image 2")
        
        self.main_fig.tight_layout()
        self.main_canvas.draw()
    
    def check_process_button(self):
        # Enable the process button if both images are selected
        if self.img1_path and self.img2_path:
            self.process_button.config(state=tk.NORMAL)
        else:
            self.process_button.config(state=tk.DISABLED)
    
    def update_alpha_label(self, value):
        # Update the alpha value display
        alpha = float(value)
        self.alpha_label.config(text=f"{alpha:.2f}")
    
    def update_frames_label(self, value):
        # Update the frames value display
        frames = int(float(value))
        self.frames_label.config(text=str(frames))
    
    def process_images(self):
        # Generate the interpolated image
        try:
            self.status_var.set("Processing... Please wait.")
            self.root.update_idletasks()
            
            alpha = self.alpha_var.get()
            self.interpolated_img = self.interpolator.interpolate(self.img1_path, self.img2_path, alpha)
            
            # Display the result
            self.ax2.clear()
            self.ax2.axis('off')
            self.ax2.imshow(self.interpolated_img, cmap='gray')
            self.ax2.set_title(f"Interpolated Result (α={alpha:.2f})")
            
            self.main_fig.tight_layout()
            self.main_canvas.draw()
            
            self.status_var.set("Interpolation complete! You can generate transition frames or save the result.")
            self.save_button.config(state=tk.NORMAL)
            self.generate_transitions_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process images: {str(e)}")
            self.status_var.set("Error occurred during processing.")
    
    def generate_transition_frames(self):
        """Generate multiple transition frames between the two images"""
        try:
            num_frames = self.frames_var.get()
            total_frames = num_frames + 2  # Include original images
            
            self.status_var.set(f"Generating {num_frames} transition frames... Please wait.")
            self.root.update_idletasks()
            
            # Reset and clear previous frames
            self.transition_frames = []
            self.current_frame_index = 0
            
            # Generate frames
            for i in range(total_frames):
                if i == 0:
                    # First reference image
                    img = cv2.imread(self.img1_path, cv2.IMREAD_GRAYSCALE)
                    self.transition_frames.append(img)
                elif i == total_frames - 1:
                    # Second reference image
                    img = cv2.imread(self.img2_path, cv2.IMREAD_GRAYSCALE)
                    self.transition_frames.append(img)
                else:
                    # Interpolated frames
                    alpha = i / (total_frames - 1)
                    img = self.interpolator.interpolate(self.img1_path, self.img2_path, alpha)
                    self.transition_frames.append(img)
                
                # Update progress
                self.status_var.set(f"Generating frames... {i+1}/{total_frames}")
                self.root.update_idletasks()
            
            # Enable navigation controls
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
            
            # Update transition display
            self.current_frame_index = 0
            self.update_transition_display()
            
            self.status_var.set(f"Generated {total_frames} transition frames. Use arrow buttons to navigate.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate transition frames: {str(e)}")
            self.status_var.set("Error occurred during frame generation.")
    
    def save_result(self):
        """Save the currently displayed result or transition frame"""
        options = ["Main Interpolation Result"]
        
        if self.transition_frames:
            options.append("Current Transition Frame")
        
        # Ask user what to save
        result = messagebox.askquestion("Save Options", "Would you like to save the main interpolation result?\n\n"
                                      "Select 'Yes' for the main result or 'No' for a transition frame.")
        
        if result == 'yes':
            # Save main interpolation result
            if self.interpolated_img is None:
                messagebox.showwarning("Warning", "No interpolated image to save.")
                return
            current_img = self.interpolated_img
        else:
            # User wants to save a transition frame
            if not self.transition_frames:
                messagebox.showwarning("Warning", "No transition frames available.")
                return
                
            # Let user select which transition frame
            transition_indices = list(range(len(self.transition_frames)))
            frame_idx = self.select_transition_frame()
            
            if frame_idx is None:
                return  # User canceled
                
            current_img = self.transition_frames[frame_idx]
        
        # Open file dialog to select save location
        filepath = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), 
                      ("TIFF files", "*.tif"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Save the image
                cv2.imwrite(filepath, current_img)
                self.status_var.set(f"Image saved successfully to {filepath}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                self.status_var.set("Error occurred while saving the image.")
    
    def select_transition_frame(self):
        """Open a dialog to let user select which transition frame to save"""
        if not self.transition_frames:
            return None
            
        # Create a simple dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Transition Frame")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select which transition frame to save:").pack(pady=(10, 5))
        
        # Frame selection dropdown
        selected_frame = tk.StringVar()
        
        frame_options = []
        for i in range(len(self.transition_frames)):
            if i == 0:
                label = "Frame 1 (Reference Image 1)"
            elif i == len(self.transition_frames) - 1:
                label = f"Frame {i+1} (Reference Image 2)"
            else:
                alpha = i / (len(self.transition_frames) - 1)
                label = f"Frame {i+1} (α={alpha:.2f})"
            frame_options.append(label)
        
        selected_frame.set(frame_options[0])
        
        frame_dropdown = ttk.Combobox(dialog, textvariable=selected_frame, values=frame_options, state="readonly", width=30)
        frame_dropdown.pack(pady=10)
        
        # Result variable
        result = [None]
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def on_cancel():
            result[0] = None
            dialog.destroy()
        
        def on_select():
            # Get the selected index from the dropdown
            selected_idx = frame_dropdown.current()
            result[0] = selected_idx
            dialog.destroy()
        
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Select", command=on_select).pack(side=tk.LEFT, padx=10)
        
        # Wait for the dialog to close
        self.root.wait_window(dialog)
        
        # Return the selected frame index
        return result[0]


def main():
    # Create main window
    root = tk.Tk()
    app = MRIInterpolationApp(root)
    
    # Set window icon if available
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "mri_icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass
    
    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()