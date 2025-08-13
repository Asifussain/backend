import pandas as pd
import traceback
from fpdf import XPos, YPos
from .base_report import BasePDFReport
from utils import sanitize_for_helvetica
from .technical_report import format_metric_for_pdf

class PatientPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "Your AI EEG Pattern Report"
        self.primary_color = (74, 144, 226)
        self.highlight_color_alz = (231, 76, 60)
        self.highlight_color_norm = (46, 204, 113)

def build_patient_pdf_report_content(pdf: PatientPDFReport, prediction_data, 
                                     similarity_data, consistency_metrics, 
                                     similarity_plot_data):
    try:
        pdf.add_page()
        
        # Section 1: Analysis Summary
        pdf.section_title("Analysis Summary")
        created_at_str = 'N/A'
        if prediction_data.get('created_at'):
            try: 
                created_at_str = pd.to_datetime(prediction_data['created_at']).strftime('%B %d, %Y')
            except: 
                created_at_str = str(prediction_data['created_at'])
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*pdf.text_color_dark)
        pdf.cell(60, 7, "File Analyzed:", 0, 0, 'L')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(*pdf.text_color_normal)
        pdf.cell(0, 7, prediction_data.get('filename', 'N/A'), 0, 1, 'L')
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*pdf.text_color_dark)
        pdf.cell(60, 7, "Date of Analysis:", 0, 0, 'L')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(*pdf.text_color_normal)
        pdf.cell(0, 7, created_at_str, 0, 1, 'L')
        pdf.ln(10)

        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(*pdf.primary_color)
        pdf.cell(0, 7, "[i] About This Report", 0, 1, 'L')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(*pdf.text_color_normal)
        pdf.ln(2)
        
        pdf.cell(0, 6, "This report uses Artificial Intelligence (AI) to analyze patterns in your brainwave (EEG) activity.", 0, 1, 'L')
        pdf.cell(0, 6, "The AI compares your EEG patterns to those learned from many examples.", 0, 1, 'L')
        pdf.ln(2)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(10, 6, "•", 0, 0, 'L')
        pdf.cell(0, 6, "IMPORTANT: This is an informational tool to help your doctor. It is NOT a medical diagnosis.", 0, 1, 'L')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(10, 6, "", 0, 0, 'L')
        pdf.cell(0, 6, "Please discuss these results with your healthcare provider.", 0, 1, 'L')
        pdf.ln(8)

        # Section 2: AI's Main Finding
        pdf.section_title("AI's Main Finding: Pattern Assessment")
        prediction_label = prediction_data.get('prediction', 'Not Determined')
        pred_display_text = "Pattern assessment inconclusive"
        pred_color = pdf.text_color_dark

        if prediction_label == "Alzheimer's":
            pred_display_text = "Patterns Suggestive of Alzheimer's Characteristics"
            pred_color = pdf.highlight_color_alz
        elif prediction_label == "Normal":
            pred_display_text = "Normal Brainwave Patterns Observed"
            pred_color = pdf.highlight_color_norm
        
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(*pdf.text_color_normal)
        pdf.cell(0, 7, "The AI analyzed your EEG and found that the patterns are most similar to:", 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(*pred_color)
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.rect(x + 10, y, pdf.w - 30, 12, 'D')
        pdf.set_xy(x + 10, y + 2)
        pdf.cell(pdf.w - 30, 8, pred_display_text, 0, 1, 'C')
        pdf.set_text_color(*pdf.text_color_normal)
        pdf.ln(8)

        probabilities = prediction_data.get('probabilities')
        confidence_text = "AI confidence score for this finding is not available."
        if isinstance(probabilities, list) and len(probabilities) == 2:
            try:
                conf_val_idx = 1 if prediction_label == "Alzheimer's" else 0
                conf_val = probabilities[conf_val_idx] * 100
                confidence_text = f"The AI is {conf_val:.0f}% confident that the patterns it found align with the finding above (based on the first segment of your EEG data)."
            except Exception as e:
                print(f"Error formatting confidence: {e}")
        
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(*pdf.primary_color)
        pdf.cell(0, 7, "[T] AI's Confidence Level", 0, 1, 'L')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(*pdf.text_color_normal)
        pdf.ln(2)
        pdf.cell(0, 6, confidence_text, 0, 1, 'L')
        pdf.ln(8)

        # Section 3: AI's Internal Consistency Check
        pdf.section_title("AI's Internal Consistency Check")
        
        if consistency_metrics and not consistency_metrics.get('error') and isinstance(consistency_metrics.get('num_trials'), int) and consistency_metrics.get('num_trials', 0) > 0:
            num_segments = consistency_metrics.get('num_trials', 'multiple')
            
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(*pdf.primary_color)
            pdf.cell(0, 7, "[M] Understanding AI's Consistency", 0, 1, 'L')
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(*pdf.text_color_normal)
            pdf.ln(2)
            
            pdf.cell(0, 6, f"To double-check its findings, the AI looked at your EEG data in {num_segments} smaller pieces (segments).", 0, 1, 'L')
            pdf.cell(0, 6, "This helps assess how stable the AI's finding was across your entire recording.", 0, 1, 'L')
            pdf.cell(0, 6, "Here's a simple breakdown:", 0, 1, 'L')
            pdf.ln(3)
            
            accuracy_val = format_metric_for_pdf(consistency_metrics.get('accuracy'), 'percent', 0)
            pdf.cell(5, 6, "•", 0, 0, 'L')
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 6, f"Overall Consistency (Accuracy): {accuracy_val}", 0, 1, 'L')
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(5, 6, "•", 0, 0, 'L')
            pdf.cell(0, 6, "This shows how often the AI's checks on the small pieces matched its main finding for your whole EEG sample.", 0, 1, 'L')
            pdf.ln(2)

            if prediction_label == "Alzheimer's":
                sensitivity_val = format_metric_for_pdf(consistency_metrics.get('recall_sensitivity'), 'percent', 0)
                precision_val = format_metric_for_pdf(consistency_metrics.get('precision'), 'percent', 0)
                f1_val = format_metric_for_pdf(consistency_metrics.get('f1_score'), 'float', 2)
                
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, f"Finding Alzheimer's-like Patterns (Sensitivity): {sensitivity_val}", 0, 1, 'L')
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.cell(0, 6, "If segments showed Alzheimer's-like patterns (based on the main finding), the AI found them this often.", 0, 1, 'L')
                pdf.ln(2)
                
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, f"Confirming Alzheimer's-like Patterns (Precision): {precision_val}", 0, 1, 'L')
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.cell(0, 6, "When the AI said a segment was Alzheimer's-like, it was consistent with the main finding this often.", 0, 1, 'L')
                pdf.ln(2)
                
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, f"Balanced Score for Alzheimer's Patterns (F1-Score): {f1_val}", 0, 1, 'L')
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.cell(0, 6, "A combined score (0 to 1, higher is better) reflecting how well the AI balanced finding and confirming these patterns.", 0, 1, 'L')
                pdf.ln(2)
            else:
                specificity_val = format_metric_for_pdf(consistency_metrics.get('specificity'), 'percent', 0)
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, f"Finding Normal Patterns (Specificity): {specificity_val}", 0, 1, 'L')
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(5, 6, "•", 0, 0, 'L')
                pdf.cell(0, 6, "If Normal patterns were present in segments (based on the main finding), the AI correctly identified them this often.", 0, 1, 'L')
                pdf.ln(2)
            
            pdf.cell(5, 6, "•", 0, 0, 'L')
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 6, f"Number of Segments Checked: {num_segments}", 0, 1, 'L')
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(5, 6, "•", 0, 0, 'L')
            pdf.cell(0, 6, "Higher percentages and scores in these checks generally suggest the AI was consistent in what it observed throughout your EEG sample.", 0, 1, 'L')
            
        elif consistency_metrics and consistency_metrics.get('message'):
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, f"Consistency check: {consistency_metrics['message']}", 0, 1, 'L')
            pdf.set_text_color(*pdf.text_color_normal)
        else:
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, "Detailed internal consistency checks were not applicable or did not yield specific metrics for this sample.", 0, 1, 'L')
            pdf.set_text_color(*pdf.text_color_normal)
        pdf.ln(10)

        # Section 4: Brainwave Shape Comparison
        if pdf.get_y() > pdf.h - 120: 
            pdf.add_page()
        
        if similarity_data and not similarity_data.get('error') and similarity_plot_data:
            plotted_ch_idx = similarity_data.get('plotted_channel_index')
            plot_title_sim = f"Comparing Your Brainwave Shape (from Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Selected'})"
            pdf.add_image_section(plot_title_sim, similarity_plot_data)

            sim_interp_text_main = "The AI found that your sample's brainwave shapes showed "
            overall_sim = similarity_data.get('overall_similarity', '')
            if "Higher Similarity to Alzheimer's Pattern" in overall_sim:
                sim_interp_text_main += "more resemblance to the Alzheimer's reference patterns."
            elif "Higher Similarity to Normal Pattern" in overall_sim:
                sim_interp_text_main += "more resemblance to the Normal reference patterns."
            else:
                sim_interp_text_main += "a mixed or inconclusive resemblance when compared to the reference patterns."
            
            sim_interpretation_from_data = similarity_data.get('interpretation', "").split("Disclaimer:")[0].replace("Similarity Analysis (DTW):", "").replace("Overall Assessment:", "").strip()
            additional_details = f"Additional Details: {sim_interpretation_from_data}" if sim_interpretation_from_data else ""
            
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(*pdf.primary_color)
            pdf.cell(0, 7, "[D] What This Graph Shows", 0, 1, 'L')
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(*pdf.text_color_normal)
            pdf.ln(2)
            pdf.cell(0, 6, sim_interp_text_main, 0, 1, 'L')
            if additional_details:
                pdf.cell(0, 6, additional_details, 0, 1, 'L')
            
        else:
            pdf.section_title("Comparing Your Brainwave Shape")
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, "The brainwave shape comparison graph is not available for this report.", 0, 1, 'L')
            pdf.set_text_color(*pdf.text_color_normal)
        pdf.ln(10)

        # Section 5: Important Information & Next Steps
        pdf.section_title("Important Information & Your Next Steps")
        
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(139, 69, 19)
        pdf.cell(0, 7, "[!] Please Discuss This Report With Your Doctor", 0, 1, 'L')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(101, 67, 33)
        pdf.ln(2)
        
        pdf.cell(0, 6, "This AI report is an informational tool based on EEG patterns. It is NOT a medical diagnosis.", 0, 1, 'L')
        pdf.ln(2)
        pdf.cell(0, 6, "Only a qualified healthcare professional can diagnose medical conditions. They will consider this", 0, 1, 'L')
        pdf.cell(0, 6, "report along with your full medical history and other tests.", 0, 1, 'L')
        pdf.ln(2)
        
        pdf.cell(5, 6, "•", 0, 0, 'L')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, f"Key Takeaway: The AI analysis suggests your EEG patterns are most similar to {sanitize_for_helvetica(pred_display_text)}.", 0, 1, 'L')
        pdf.ln(2)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(5, 6, "•", 0, 0, 'L')
        pdf.cell(0, 6, "Recommended Next Steps:", 0, 1, 'L')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(15, 6, "◦", 0, 0, 'L')
        pdf.cell(0, 6, "Share this entire report with your doctor or a neurologist.", 0, 1, 'L')
        pdf.cell(15, 6, "◦", 0, 0, 'L')
        pdf.cell(0, 6, "Discuss any health concerns and follow their medical advice.", 0, 1, 'L')
        pdf.cell(15, 6, "◦", 0, 0, 'L')
        pdf.cell(0, 6, "Ask your doctor to explain what these findings mean in the context of your overall health.", 0, 1, 'L')
        
        pdf.set_text_color(*pdf.text_color_normal)
        
    except Exception as e:
        print(f"Error building Patient PDF content: {e}")
        traceback.print_exc()
        try:
            if pdf.page_no() == 0: 
                pdf.add_page()
            elif pdf.get_y() > pdf.h - 30: 
                pdf.add_page()
            
            pdf.set_font("Helvetica", 'B', 12)
            pdf.set_text_color(255, 0, 0)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.rect(x + 5, y, pdf.w - 20, 25, 'D')
            pdf.set_xy(x + 10, y + 5)
            pdf.cell(pdf.w - 30, 8, "Critical Error Building PDF Content:", 0, 1, 'C')
            pdf.set_xy(x + 10, y + 12)
            pdf.set_font("Helvetica", '', 10)
            pdf.cell(pdf.w - 30, 8, sanitize_for_helvetica(str(e)), 0, 1, 'C')
            pdf.set_text_color(*pdf.text_color_normal)
        except Exception as pdf_err_fallback:
            print(f"Fallback error writing to Patient PDF failed: {pdf_err_fallback}")
