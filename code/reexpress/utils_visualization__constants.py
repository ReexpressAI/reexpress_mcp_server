# Copyright Reexpress AI, Inc. All rights reserved.

css_style = f"""
    <style>
        body {{
            background-color: #f5f5f5;
            color: #212529;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}

        .header {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #1a1a1a;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section {{
            margin-bottom: 25px;
        }}

        .section-title {{
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .field-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .field-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 12px 16px;
        }}

        .field-label {{
            font-size: 13px;
            color: #6c757d;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 5px;
            font-weight: 500;
        }}

        .field-value {{
            font-size: 15px;
            color: #212529;
            font-weight: 600;
        }}

        .icon {{
            width: 16px;
            height: 16px;
            display: inline-block;
            vertical-align: middle;
        }}

        .tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 5px;
            font-weight: 600;
        }}

        .tag-positive {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .tag-caution {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }}
        
        .tag-negative {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}

        .tag-neutral {{
            background-color: #e2e3e5;
            color: #383d41;
            border: 1px solid #d6d8db;
        }}

        .tag-highest {{
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}

        .prompt-box, .document-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 16px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 14px;
            line-height: 1.5;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #212529;
        }}
        
        .model-name {{
            font-family: "Consolas", "Monaco", monospace;
            font-size: 12px;
            color: #6c757d;
            font-weight: 300;
        }}
        
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px solid #ffeaa7;
        }}

        .resize-button {{
            float: right;
            background: none;
            border: none;
            color: #0056b3;
            cursor: pointer;
            font-size: 14px;
            padding: 4px 8px;
            font-weight: 500;
        }}

        .resize-button:hover {{
            color: #004085;
            text-decoration: underline;
        }}
        
        .nearest-match-box {{
            border-left: 4px solid #dee2e6;
        }}
        
        .explanation-box {{
            background-color: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 0 6px 6px 0;
        }}

        .explanation-title {{
            font-weight: 700;
            color: #004085;
            margin-bottom: 4px;
        }}
        
        .explanation-box-positive {{
            background-color: #f8f9fa;                        
            border: 1px solid #dee2e6;
            border-left: 4px solid #155724;
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 0 6px 6px 0;
        }}

        .explanation-title-positive {{
            font-weight: 700;
            color: #155724;
            margin-bottom: 4px;
        }}
        
        .explanation-box-negative {{
            background-color: #f8f9fa;                        
            border: 1px solid #dee2e6;
            border-left: 4px solid #721c24;
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 0 6px 6px 0;
        }}

        .explanation-title-negative {{
            font-weight: 700;
            color: #721c24;
            margin-bottom: 4px;
        }}

        .separator {{
            border-top: 2px solid #dee2e6;
            margin: 30px 0;
        }}

        .info-icon {{
            color: #0056b3;
        }}

        .checkmark {{
            color: #28a745;
            font-weight: bold;
        }}

        .cross {{
            color: #dc3545;
            font-weight: bold;
        }}

        .math-operator {{
            font-family: 'Times New Roman', serif;
            font-style: italic;
            display: inline-block;
        }}
        
        .math-tilde {{
            position: relative;
        }}
        
        .math-superscript {{
            position: absolute;
            top: -0.5em;
            right: -0.2em;
            font-size: 0.75em;
        }}
        
        .math-subscript {{
            font-size: 0.75em;
            font-style: normal;
            vertical-align: sub;
        }}
        
        .math-qtilde {{
            display: inline-block;
            position: relative;
            font-style: italic;
            width: 0.7em;
            text-align: center;
        }}
        
        .math-qtilde::after {{
            content: "~";
            position: absolute;
            top: -0.3em;
            left: 0;
            right: 0;
            font-style: normal;
        }}
        
        .math-operator-m {{
            font-family: 'Times New Roman', serif;
            display: inline-block;
            position: relative;
            padding-right: 1.2em; /* Space for super/subscripts */
        }}
        
        .math-superscript-hat-y {{
            position: absolute;
            top: -0.5em;
            left: 1.2em;
            font-size: 0.75em;
        }}
        
        .math-subscript-floor {{
            position: absolute;
            bottom: -0.3em;
            left: 1.2em;
            font-size: 0.75em;
            white-space: nowrap;
        }}
        
        .qtilde-small {{
            display: inline-block;
            position: relative;
            width: 0.6em;
            font-style: italic;
        }}
        
        .qtilde-small::after {{
            content: "~";
            position: absolute;
            top: -0.2em;
            left: 0;
            right: 0;
            text-align: center;
            font-style: normal;
            font-size: 0.9em;
        }}
        
        .math-parens {{
            font-family: 'Times New Roman', serif;
        }}
        
        .math-parens > .paren {{
            font-size: 1.3em;
            vertical-align: -0.1em;
        }}

        /* For Legend */

        .legend-content {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        
        .legend-content p {{
            margin-bottom: 15px;
            line-height: 1.6;
            color: #555;
        }}
        
        .legend-items {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 8px 12px;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        
        .legend-label {{
            font-weight: 600;
            color: #333;
            margin-right: 10px;
            min-width: 60px;
        }}
        
        .legend-value {{
            color: #666;
            font-family: 'Courier New', monospace;
        }}
        
    </style>
"""
