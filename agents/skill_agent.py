# agents/skill_agent.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, List, Optional, Any

from utils.skill_loader import SkillLoader
from agents.data_science_agent import EnhancedDataScienceAgent
import pandas as pd
import json

class SkillBasedAgent:
    """
    Agent that uses Anthropic-style skills to guide data science tasks
    """
    
    def __init__(self, df: pd.DataFrame, azure_client=None, filename=None):
        self.df = df
        self.azure_client = azure_client
        self.filename = filename  # Store filename
        self.data_agent = EnhancedDataScienceAgent(df, azure_client)
        
        # ===== FIX: Set filename in data_agent =====
        if filename:
            self.data_agent.filename = filename
            print(f"📁 Filename set in data_agent: {filename}")
        
        self.skill_loader = SkillLoader()
        self.current_skill = None
        self.skill_context = {}

  
    def detect_and_apply_skill(self, prompt: str, task_type: str = None):
        """
        Detect appropriate skill and apply it
        """
        print(f"\n🎯 Detecting skill for: {prompt}")
        
        # Step 1: If task_type not provided, detect it
        if not task_type and self.azure_client:
            task_info = self.data_agent.detect_task_type(prompt)
            task_type = task_info.get('task_type', 'exploratory')
            print(f"   Detected task type: {task_type}")
        
        # Step 2: Find relevant skills
        relevant_skills = self.skill_loader.find_skill_for_task(task_type, prompt)
        
        if not relevant_skills:
            print("   No specific skill found, using default approach")
            return self._execute_without_skill(prompt, task_type)
        
        # Step 3: Select best skill (first one for now)
        selected_skill = relevant_skills[0]
        skill_name = selected_skill['name']
        print(f"   Selected skill: {skill_name}")
        
        self.current_skill = skill_name
        
        # Step 4: Prepare context for skill
        context = {
            'dataset_shape': self.df.shape,
            'columns': list(self.df.columns),
            'numeric_cols': self.data_agent.numeric_cols,
            'categorical_cols': self.data_agent.categorical_cols,
            'task_type': task_type,
            'user_prompt': prompt,
            'total_rows': len(self.df)
        }
        
        # Step 5: Get skill instructions
        instructions = self.skill_loader.apply_skill_instructions(skill_name, context)
        print(f"\n📋 Skill Instructions:\n{instructions[:500]}...")
        
        # Step 6: Execute based on task type
        return self._execute_skill_task(task_type, prompt, instructions)
    
    # agents/skill_agent.py - Fix target detection and prompt handling

    def _execute_skill_task(self, task_type: str, prompt: str, instructions: str):
        """
        Execute the actual task based on skill guidance
        """
        print(f"\n🚀 Executing {task_type} task with skill guidance...")
        
        # Store skill context
        self.skill_context = {
            'skill': self.current_skill,
            'instructions': instructions,
            'task_type': task_type
        }
        
        # Execute using data_agent
        if task_type == 'regression':
            target_col = None
            
            # Try to detect from prompt FIRST (if prompt exists)
            if prompt and prompt != 'null' and self.azure_client:
                try:
                    task_info = self.data_agent.detect_task_type(prompt)
                    if task_info.get('target_column'):
                        target_col = task_info['target_column']
                        target_clean = target_col.strip('"').strip("'").strip()
                        
                        # Check if target exists in dataframe
                        if target_clean not in self.df.columns:
                            found = False
                            for col in self.df.columns:
                                col_clean = col.strip('"').strip("'").strip()
                                if col_clean.lower() == target_clean.lower():
                                    target_col = col
                                    print(f"✅ Found match: {col}")
                                    found = True
                                    break
                            
                            if not found:
                                print(f"⚠️ Target '{target_clean}' not found")
                                target_col = None
                        else:
                            print(f"   🎯 Target from AI prompt: {target_clean}")
                except Exception as e:
                    print(f"   ⚠️ AI detection failed: {e}")
            
            # If not found, look for 'price' column
            if not target_col:
                for col in self.data_agent.numeric_cols:
                    if 'price' in col.lower() or 'prize' in col.lower():
                        target_col = col
                        print(f"   🎯 Found price column: {target_col}")
                        break
            
            # If still not found, use last numeric column
            if not target_col and len(self.data_agent.numeric_cols) > 0:
                target_col = self.data_agent.numeric_cols[-1]
                print(f"   🎯 Auto-selected last numeric: {target_col}")
            
            if target_col:
                return self.data_agent.perform_regression(target_col)
            else:
                return {"error": "No numeric column for regression"}



        elif task_type == 'classification':
            # ========== FIX 3: Similar validation for classification ==========
            target_col = None
            
            # Try to detect from prompt
            if prompt and prompt != 'null' and self.azure_client:
                try:
                    task_info = self.data_agent.detect_task_type(prompt)
                    if task_info.get('target_column'):
                        target_col = task_info['target_column']
                        print(f"   🎯 Target from AI prompt: {target_col}")
                except Exception as e:
                    print(f"   ⚠️ AI detection failed: {e}")
            
            # If not found, use first categorical column
            if not target_col and len(self.data_agent.categorical_cols) > 0:
                target_col = self.data_agent.categorical_cols[0]
                print(f"   🎯 Auto-selected categorical: {target_col}")
            
            # ========== ✅ Validate target column exists ==========
            if target_col:
                # Check if target_col exists in dataframe (with cleaning)
                if target_col not in self.df.columns:
                    print(f"   ⚠️ Target '{target_col}' not found in columns")
                    print(f"   📋 Available columns: {list(self.df.columns)}")
                    
                    # Try to find matching column
                    target_clean = target_col.strip('"').strip()
                    found = False
                    for col in self.df.columns:
                        col_clean = col.strip('"').strip()
                        if col_clean.lower() == target_clean.lower():
                            target_col = col
                            print(f"   ✅ Found match: {col}")
                            found = True
                            break
                    
                    if not found and len(self.data_agent.categorical_cols) > 0:
                        target_col = self.data_agent.categorical_cols[0]
                        print(f"   ⚠️ Using fallback column: {target_col}")
                
                return self.data_agent.perform_classification(target_col)
            else:
                return {"error": "No categorical column for classification"}
        
        elif task_type == 'clustering':
            return self.data_agent.perform_clustering()
        
        else:  # exploratory/visualization
            # Only do visualization if prompt exists
            if prompt and prompt != 'null':
                visualizations = self.data_agent.perform_enhanced_visualization(prompt)
            else:
                # If no prompt, do basic analysis
                print("   No prompt provided, doing basic data analysis")
                visualizations = []
                visualizations.extend(self.data_agent.create_correlation_analysis())
                visualizations.extend(self.data_agent.create_distribution_analysis())
            
            return {
                'message': 'Analysis completed',
                'visualization_count': len(visualizations),
                'visualizations': visualizations[:5]
            }    
        


    def _execute_without_skill(self, prompt: str, task_type: str):
        """
        Fallback when no skill found
        """
        print("   Using fallback execution")
        
        if task_type in ['regression', 'classification', 'clustering']:
            return self._execute_skill_task(task_type, prompt, "")
        else:
            visualizations = self.data_agent.perform_enhanced_visualization(prompt)
            return {
                'message': 'Exploratory analysis completed',
                'visualization_count': len(visualizations),
                'visualizations': visualizations[:5]
            }
    
    def get_skill_info(self) -> Dict[str, Any]:
        """Get current skill information"""
        return {
            'current_skill': self.current_skill,
            'context': self.skill_context,
            'available_skills': self.skill_loader.list_all_skills()
        }
    
    def list_all_skills(self) -> List[Dict[str, Any]]:
        """List all available skills"""
        return self.skill_loader.list_all_skills()