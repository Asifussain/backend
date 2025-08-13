import base64
import io
import pandas as pd
import traceback
from .base_report import BasePDFReport
from utils import sanitize_for_helvetica
from .technical_report import format_metric_for_pdf

class ClinicianPDFReport(BasePDFReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = "AI-Assisted EEG Pattern Analysis Report"
        self.primary_color = (41, 128, 185)
        self.secondary_color = (52, 152, 219)

def build_clinician_pdf_report_content(pdf: ClinicianPDFReport, prediction_data, stats_data, 
                                       similarity_data, consistency_metrics, 
                                       ts_img_data, psd_img_data, similarity_plot_data):
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    try:
        pdf.add_page()

        pdf.section_title("Patient & Analysis Overview")
        pdf.key_value_pair("Patient Identifier/Filename", prediction_data.get('filename', 'N/A'))
        
        created_at = prediction_data.get('created_at')
        date_str = 'N/A'
        if created_at:
            try:
                dt_obj = pd.to_datetime(created_at)
                date_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S UTC') if dt_obj.tzinfo else dt_obj.strftime('%Y-%m-%d %H:%M:%S (Unknown TZ)')
            except Exception:
                date_str = str(created_at)
        pdf.key_value_pair("Date of Analysis", date_str)

        prediction_label = prediction_data.get('prediction', 'N/A')
        if prediction_label == "Alzheimer's":
            primary_finding = "Pattern Suggestive of Alzheimer's-related Changes"
        elif prediction_label == "Normal":
            primary_finding = "Normal EEG Pattern"
        else:
            primary_finding = "Indeterminate"
        pdf.key_value_pair("Primary AI Finding (EEG Pattern)", primary_finding)

        probabilities = prediction_data.get('probabilities')
        prob_str = 'N/A'
        if isinstance(probabilities, list) and len(probabilities) == 2:
            try:
                prob_str = f"Normal: {format_metric_for_pdf(probabilities[0], 'percent')}, Alzheimer's-related Changes: {format_metric_for_pdf(probabilities[1], 'percent')}"
            except Exception:
                prob_str = sanitize_for_helvetica(str(probabilities))
        elif probabilities is not None:
            prob_str = sanitize_for_helvetica(str(probabilities))
        pdf.key_value_pair("Assessment Confidence", prob_str)
        pdf.ln(5)

        pdf.section_title("EEG Pattern Analysis & Consistency")
        consistency_statement = "(Consistency check not available or not applicable for this recording.)"
        if consistency_metrics and not consistency_metrics.get('error'):
            num_trials = consistency_metrics.get('num_trials', 0)
            if num_trials > 1:
                accuracy = consistency_metrics.get('accuracy', 0)
                if accuracy >= 0.85:
                    consistency_statement = f"High (Overall segment agreement: {format_metric_for_pdf(accuracy, 'percent',0)} across {num_trials} segments)"
                elif accuracy >= 0.70:
                    consistency_statement = f"Moderate (Overall segment agreement: {format_metric_for_pdf(accuracy, 'percent',0)} across {num_trials} segments)"
                else:
                    consistency_statement = f"Low (Overall segment agreement: {format_metric_for_pdf(accuracy, 'percent',0)} across {num_trials} segments. Interpret with caution.)"
            elif consistency_metrics.get('message'):
                consistency_statement = f"({consistency_metrics.get('message')})"
        elif consistency_metrics and consistency_metrics.get('error'):
            consistency_statement = "(Could not determine finding consistency due to an analysis error.)"
        pdf.key_value_pair("AI Finding Consistency", consistency_statement)
        pdf.ln(10)

        if pdf.get_y() > pdf.h - 80:
            pdf.add_page()

        pdf.section_title("Key EEG Waveform Characteristics")
        if similarity_data and not similarity_data.get('error'):
            interpretation = similarity_data.get('interpretation', 'No specific waveform characteristics noted or interpretation available from DTW analysis.')
            interpretation_clean = interpretation.split("Disclaimer:")[0].replace("Similarity Analysis (DTW):", "").replace("Overall Assessment:", "").strip()
            pdf.write_multiline(interpretation_clean if interpretation_clean else "No interpretation available.", indent=5)
            pdf.ln(2)
            if similarity_plot_data:
                plotted_ch_idx = similarity_data.get('plotted_channel_index')
                plot_title_sim_clin = f"Illustrative EEG Segment Comparison (Channel {plotted_ch_idx + 1 if plotted_ch_idx is not None else 'Selected'})"
                pdf.add_image_section(plot_title_sim_clin, similarity_plot_data)
            else:
                pdf.set_font("Helvetica",'I',10)
                pdf.cell(0,10,"(No illustrative segment comparison plot available.)", ln=1)
        else:
            err_msg_sim_clin = "(Waveform characteristics analysis (DTW) not performed or encountered an error.)"
            pdf.set_font("Helvetica",'I',10)
            pdf.write_multiline(err_msg_sim_clin, indent=5)
        pdf.ln(10)

        if pdf.get_y() > pdf.h - 100:
            pdf.add_page()

        pdf.section_title("Brainwave Frequency Profile")
        if stats_data and not stats_data.get('error') and stats_data.get('avg_band_power'):
            pdf.set_font("Helvetica",'B',11)
            pdf.cell(0,6,"Relative Brainwave Activity (%):", ln=1)
            pdf.ln(1)
            pdf.set_font("Helvetica",'',10)
            avg_power = stats_data.get('avg_band_power',{})
            band_data_points = []
            for band, powers in avg_power.items():
                rel_power_str = format_metric_for_pdf(powers.get('relative'), 'percent', 1)
                band_data_points.append(f"{band.capitalize()}: {rel_power_str}")
            if band_data_points:
                for point in band_data_points:
                    pdf.cell(10)
                    pdf.cell(0,5,point, ln=1)
            else:
                pdf.set_font("Helvetica",'I',10)
                pdf.cell(10)
                pdf.cell(0,5,"(Detailed brainwave activity data not available.)", ln=1)
        else:
            err_msg_stats_clin = "(Frequency profile analysis not performed or encountered an error.)"
            pdf.set_font("Helvetica",'I',10)
            pdf.write_multiline(err_msg_stats_clin, indent=5)
        pdf.ln(10)

        if pdf.get_y() > pdf.h - 120:
            pdf.add_page()

        pdf.section_title("EEG Visualizations (Reference)")
        pdf.add_image_section("Selected EEG Traces", ts_img_data)
        
        if pdf.get_y() > pdf.h - 120:
            pdf.add_page()
        pdf.add_image_section("Overall Frequency Spectrum (PSD)", psd_img_data)
        pdf.ln(5)

        if pdf.get_y() > pdf.h - 50:
            pdf.add_page()
        pdf.section_title("Clinical Considerations & Disclaimer")
        disclaimer_text_clinician = (
            "This AI-assisted report is intended to supplement clinical judgment, not replace it. "
            "The findings are based on algorithmic pattern recognition in EEG data and should be "
            "correlated with comprehensive clinical evaluation, patient history, and other diagnostic results. "
            "This report does not constitute a medical diagnosis."
        )
        pdf.set_font("Helvetica",'',9)
        pdf.set_text_color(80,80,80)
        pdf.write_multiline(disclaimer_text_clinician, indent=0)
        pdf.set_text_color(*pdf.text_color_normal)
        pdf.ln(5)

    except Exception as pdf_build_e:
        print(f"Critical Error building Clinician PDF content: {pdf_build_e}")
        traceback.print_exc()
        try:
            if pdf.page_no() == 0:
                pdf.add_page()
            elif pdf.get_y() > pdf.h - 30:
                pdf.add_page()
            pdf.set_font("Helvetica",'B',12)
            pdf.set_text_color(255,0,0)
            pdf.multi_cell(0,10,f"Critical Error Building PDF Content:\n{sanitize_for_helvetica(str(pdf_build_e))}",align='C')
            pdf.set_text_color(*pdf.text_color_normal)
        except Exception as pdf_err_fallback:
            print(f"Fallback error writing critical error to Clinician PDF failed: {pdf_err_fallback}")