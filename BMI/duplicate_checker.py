from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import hashlib
import os
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'duplicate_uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv', 'xls'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_line(value):
    """Clean data from row - remove spaces and extract comma-separated parts"""
    if pd.isna(value):
        return None
    
    val_str = "".join(str(value).split())
    parts = val_str.split(",")
    if len(parts) >= 5:
        return ",".join(parts[2:-1])
    return None


def process_file(filepath, filename):
    """Process a single file and return hash and data info"""
    try:
        # Read the file
        if filename.endswith('.csv'):
            df_check = pd.read_csv(filepath)
        else:  # xlsx or xls
            try:
                df_check = pd.read_excel(filepath, engine='openpyxl')
            except Exception as xlsx_error:
                # Try with xlrd engine as fallback
                try:
                    df_check = pd.read_excel(filepath, engine='xlrd')
                except:
                    # If both fail, return error with specific message
                    error_msg = str(xlsx_error)
                    if "not a zip file" in error_msg or "ZIP file" in error_msg:
                        return {
                            'success': False,
                            'filename': filename,
                            'reason': f'ไฟล์ .xlsx ถูกเสียหาย หรือไม่ใช่ไฟล์ Excel ที่ถูกต้อง',
                            'hash': None
                        }
                    else:
                        return {
                            'success': False,
                            'filename': filename,
                            'reason': f'Error: {error_msg[:100]}',
                            'hash': None
                        }
        
        # Check if file is empty
        if df_check.empty:
            return {
                'success': False,
                'filename': filename,
                'reason': 'ไฟล์ว่างเปล่า',
                'hash': None
            }
        
        # Check if has at least one column
        if len(df_check.columns) < 1:
            return {
                'success': False,
                'filename': filename,
                'reason': 'ไม่มีคอลัมน์ข้อมูล',
                'hash': None
            }
        
        # Process first column
        col = df_check.iloc[:, 0].dropna().astype(str)
        col_cleaned = col.apply(clean_line).dropna()
        
        # Check if data format is correct
        if col_cleaned.empty:
            return {
                'success': False,
                'filename': filename,
                'reason': 'รูปแบบข้อมูล (Comma) ไม่ถูกต้อง หรือไม่มีข้อมูลที่ตรงเงื่อนไข',
                'hash': None
            }
        
        # Generate hash
        col_sorted = col_cleaned.sort_values()
        combined = "\n".join(col_sorted)
        file_hash = hashlib.md5(combined.encode("utf-8")).hexdigest()
        
        return {
            'success': True,
            'filename': filename,
            'hash': file_hash,
            'row_count': len(col_cleaned),
            'reason': None
        }
        
    except Exception as e:
        error_msg = str(e)
        if "not a zip file" in error_msg or "ZIP file" in error_msg:
            return {
                'success': False,
                'filename': filename,
                'reason': f'ไฟล์ .xlsx ถูกเสียหาย หรือไม่ใช่ไฟล์ Excel ที่ถูกต้อง',
                'hash': None
            }
        return {
            'success': False,
            'filename': filename,
            'reason': f'Error: {error_msg[:100]}',
            'hash': None
        }


@app.route('/')
def index():
    """Render the main page"""
    return render_template('duplicate_checker.html')


@app.route('/api/process-folder', methods=['POST'])
def process_folder():
    """Handle folder processing - scan directory for duplicate files"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '')
        
        if not folder_path or not os.path.exists(folder_path):
            return jsonify({'success': False, 'error': 'Folder path is invalid'}), 400
        
        if not os.path.isdir(folder_path):
            return jsonify({'success': False, 'error': 'Path is not a folder'}), 400
        
        file_results = []
        bad_files = []
        
        # Scan folder for allowed files
        for filename in os.listdir(folder_path):
            if not allowed_file(filename) or filename.startswith('~$'):
                continue
            
            filepath = os.path.join(folder_path, filename)
            
            if os.path.isfile(filepath):
                # Process file
                result = process_file(filepath, filename)
                
                if result['success']:
                    file_results.append(result)
                else:
                    bad_files.append({
                        'filename': result['filename'],
                        'reason': result['reason']
                    })
        
        if not file_results:
            return jsonify({
                'success': False,
                'error': 'No valid files found in folder'
            }), 400
        
        # Find duplicates
        duplicates = []
        hash_groups = {}
        
        for file_info in file_results:
            file_hash = file_info['hash']
            if file_hash not in hash_groups:
                hash_groups[file_hash] = []
            hash_groups[file_hash].append(file_info)
        
        # Get only duplicate groups
        for file_hash, group in hash_groups.items():
            if len(group) > 1:
                duplicates.append({
                    'hash': file_hash,
                    'files': group,
                    'count': len(group)
                })
        
        return jsonify({
            'success': True,
            'total_files': len(file_results) + len(bad_files),
            'processed_files': len(file_results),
            'bad_files': len(bad_files),
            'duplicate_groups': len(duplicates),
            'file_results': file_results,
            'duplicates': duplicates,
            'bad_files_list': bad_files,
            'folder_path': folder_path
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing folder: {str(e)}'}), 500


@app.route('/api/browse-folders', methods=['GET'])
def browse_folders():
    """Get list of drives or recent folders on Windows"""
    try:
        import string
        folders = []
        
        # Get all drives on Windows
        for drive in string.ascii_uppercase:
            path = f"{drive}:\\"
            if os.path.exists(path):
                folders.append({
                    'name': drive + ':',
                    'path': path,
                    'type': 'drive'
                })
        
        # Add Desktop
        desktop_path = os.path.expanduser('~/Desktop')
        if os.path.exists(desktop_path):
            folders.append({
                'name': 'Desktop',
                'path': desktop_path,
                'type': 'folder'
            })
        
        return jsonify({
            'success': True,
            'folders': folders
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze-location', methods=['POST'])
def analyze_location():
    """Analyze location data from folder - extract and group location data"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '')
        
        if not folder_path or not os.path.exists(folder_path):
            return jsonify({'success': False, 'error': 'Folder path is invalid'}), 400
        
        if not os.path.isdir(folder_path):
            return jsonify({'success': False, 'error': 'Path is not a folder'}), 400
        
        all_data = []
        bad_files = []
        
        # Scan folder for allowed files
        for filename in os.listdir(folder_path):
            if not allowed_file(filename) or filename.startswith('~$'):
                continue
            
            filepath = os.path.join(folder_path, filename)
            
            if os.path.isfile(filepath):
                try:
                    # Read the file - only first column
                    if filename.endswith('.csv'):
                        df = pd.read_csv(filepath, usecols=[0])
                    else:  # xlsx or xls
                        df = pd.read_excel(filepath, usecols=[0], engine='openpyxl')
                    
                    if df.empty:
                        bad_files.append({'filename': filename, 'reason': 'ไฟล์ว่างเปล่า'})
                        continue
                    
                    # Drop NaN values
                    col = df.iloc[:, 0].dropna()
                    
                    if col.empty:
                        bad_files.append({'filename': filename, 'reason': 'ไม่มีข้อมูลหลังลบ NaN'})
                        continue
                    
                    # Add filename column
                    col_df = pd.DataFrame({
                        'data': col.astype(str),
                        'filename': filename
                    })
                    all_data.append(col_df)
                    
                except Exception as e:
                    bad_files.append({'filename': filename, 'reason': f'Error: {str(e)}'})
        
        if not all_data:
            return jsonify({
                'success': False,
                'error': 'No valid data found in folder'
            }), 400
        
        # Combine all data
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Split by comma
        split_data = final_df['data'].str.split(',', expand=True)
        
        # Create location dataframe with split columns and filename
        location_df = pd.concat([split_data, final_df['filename']], axis=1)
        
        # Group by column 1 (second column after split)
        # Column index 1 contains location data (parts[2] from original)
        if 1 not in location_df.columns:
            return jsonify({
                'success': False,
                'error': 'Invalid data format - cannot find location column'
            }), 400
        
        # Group and count
        groupby_result = location_df.groupby(1)[location_df.columns[-1]].value_counts().reset_index(name='count')
        groupby_result.columns = ['Location', 'Filename', 'Count']
        
        # Convert to list of dicts for JSON
        result_list = groupby_result.to_dict('records')
        
        # Also get summary statistics
        location_summary = location_df.groupby(1)[location_df.columns[-1]].nunique().reset_index()
        location_summary.columns = ['Location', 'Unique_Files']
        location_summary_list = location_summary.to_dict('records')
        
        # Get unique locations
        unique_locations = sorted(location_df[1].dropna().unique().tolist())
        
        return jsonify({
            'success': True,
            'total_rows': len(final_df),
            'total_files': len(all_data),
            'bad_files': len(bad_files),
            'unique_locations': len(unique_locations),
            'locations_list': unique_locations,
            'groupby_data': result_list,
            'summary_data': location_summary_list,
            'bad_files_list': bad_files,
            'folder_path': folder_path
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error analyzing location: {str(e)}'}), 500


@app.route('/api/upload-files', methods=['POST'])
def upload_files():
    """Handle multiple file uploads and check for duplicates"""
    
    # Check if files are in request
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    try:
        file_results = []
        bad_files = []
        
        # Process each file
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                bad_files.append({
                    'filename': file.filename,
                    'reason': 'Invalid file type. Please upload .xlsx, .xls, or .csv file'
                })
                continue
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process file
            result = process_file(filepath, filename)
            
            if result['success']:
                file_results.append(result)
            else:
                bad_files.append({
                    'filename': result['filename'],
                    'reason': result['reason']
                })
        
        # Find duplicates
        duplicates = []
        hash_groups = {}
        
        for file_info in file_results:
            file_hash = file_info['hash']
            if file_hash not in hash_groups:
                hash_groups[file_hash] = []
            hash_groups[file_hash].append(file_info)
        
        # Get only duplicate groups
        for file_hash, group in hash_groups.items():
            if len(group) > 1:
                duplicates.append({
                    'hash': file_hash,
                    'files': group,
                    'count': len(group)
                })
        
        return jsonify({
            'success': True,
            'total_files': len(files),
            'processed_files': len(file_results),
            'bad_files': len(bad_files),
            'duplicate_groups': len(duplicates),
            'file_results': file_results,
            'duplicates': duplicates,
            'bad_files_list': bad_files
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing files: {str(e)}'}), 500


import threading
import webbrowser
from time import sleep

def open_browser():
    sleep(1.5)
    webbrowser.open("http://localhost:5001/")

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=False, host='localhost', port=5001)
