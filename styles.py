# styles.py
import streamlit as st

def load_global_css():
    st.markdown("""
        <style>
            /* --- Hide Default Streamlit Hamburger Menu and Footer --- */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* header {visibility: hidden;} */ /* Keep Streamlit's header for MPA page names */

            /* --- General Page Adjustments --- */
            .block-container { /* Reduce default padding of the main content area */
                padding-top: 1.5rem; 
                padding-bottom: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }

            /* --- Metric Card Styles --- */
            .stMetric {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 15px 18px !important; /* Specific padding for metric content */
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                transition: transform .2s;
                height: 135px; /* Fixed height for alignment across columns */
                display: flex;
                flex-direction: column;
                justify-content: center; /* Center content vertically */
            }
            .stMetric:hover {
                transform: scale(1.02);
            }
            
            /* Targeting the label, value, and delta specifically for st.metric */
            .stMetric > label[data-testid="stMetricLabel"] { 
                font-weight: 500 !important;
                color: #4F4F4F !important;
                font-size: 0.9rem !important; /* Adjusted for readability */
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis; /* Add ellipsis if label is too long */
                margin-bottom: 5px !important; /* Space between label and value */
            }
            .stMetric .stMetricValue { 
                font-size: 1.7rem !important; /* Adjusted for prominence */
                font-weight: 600 !important;
                color: #1E88E5 !important; /* Example primary color for value */
                line-height: 1.2 !important;
            }
            .stMetric .stMetricDelta { 
                font-size: 0.85rem !important; 
                font-weight: 500 !important;
                color: #555 !important; /* Default delta color, can be overridden by Streamlit for positive/negative */
            }
            /* Help text styling for st.metric (if you use the help parameter) */
            .stMetric div[data-testid="stMetricHelp"] {
                font-size: 0.75rem !important;
                color: #6c757d !important;
            }


            /* --- Section Divider --- */
            hr.section-divider {
                margin-top: 2rem;
                margin-bottom: 2rem;
                border: 0;
                border-top: 2px solid #e0e0e0;
            }
            /* Standard hr for navigation and other separators */
            hr {
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }


            /* --- AI Analysis Report Styles (Common Elements) --- */
            .report-header {text-align: center; margin-bottom: 30px;}
            .report-section {
                margin-bottom: 30px; /* Slightly reduced margin */
                padding: 20px; 
                background-color: #f9f9f9; 
                border-radius: 8px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.08); /* Softer shadow */
            }
            .report-section h2 { 
                color: #004085; 
                border-bottom: 2px solid #b0c4de; /* Lighter border color */
                padding-bottom: 10px; 
                margin-top: 0px; /* Remove top margin if it's the first element */
                margin-bottom:20px; 
                font-size: 1.6em; /* Slightly reduced */
            }
            .report-section h3 { 
                color: #333; 
                margin-top: 20px; 
                margin-bottom:10px; 
                font-size: 1.3em; /* Slightly reduced */
            }
            .metric-card-ai { /* Used in AI report for specific cards */
                text-align: center; 
                padding: 15px; 
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                background-color: #fff;
                min-height: 90px; /* Ensure some min height */
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .metric-card-ai .label {font-size: 0.9em; color: #555; margin-bottom: 5px;}
            .metric-card-ai .value {font-size: 1.4em; font-weight: bold; color: #007bff;} /* Reduced value size */
            
            .explanation-text {
                font-style: italic; 
                color: #555; 
                margin-top: 8px; /* Adjusted */
                margin-bottom:15px; 
                line-height: 1.5; /* Adjusted */
                font-size: 0.9em;
            }
            .list-item { margin-bottom: 8px; list-style-position: inside; padding-left: 5px;}
            .key-item { margin-bottom: 10px; } /* For AI report key positives/negatives */
            .positive-item { color: #28a745; font-weight: bold;}
            .negative-item { color: #dc3545; font-weight: bold;}
            .neutral-item { color: #6c757d; }

            /* Table styles for manually generated HTML tables */
            table {
                border-collapse: collapse;
                width: 100%;
                font-size: 0.9rem; /* Base font size for tables */
            }
            th, td {
                text-align: left;
                padding: 8px 10px; /* Consistent padding */
                border-bottom: 1px solid #ddd; /* Default border for all cells */
            }
            th { /* Table headers */
                background-color: #f8f9fa;
                font-weight: 600; /* Bolder headers */
                color: #333;
            }
            /* Specific for metric name column in financial tables */
            td:first-child { 
                font-weight: 500; /* Make metric names slightly bolder than values */
                 /* color: #004085; */ /* Optional: color for metric names */
            } 
            tr:hover {
                background-color: #f1f1f1; /* Hover effect for table rows */
            }
            /* Category headers in financial tables */
            .table-category-header {
                font-weight: bold;
                font-size: 1.05em;
                padding: 10px 5px 8px 5px; /* Adjusted padding */
                background-color: #e9ecef; /* Slightly darker than th for distinction */
                border-bottom: 2px solid #ced4da !important; /* Stronger border for category */
                color: #212529;
            }
            .table-metric-name {
                 padding-left: 15px !important; /* Indent metric names under category */
            }
             .table-value-cell {
                text-align: right !important; /* Ensure values are right-aligned */
            }


        </style>
    """, unsafe_allow_html=True)