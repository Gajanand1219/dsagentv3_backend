# utils/skill_loader.py
import os
import json
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class SkillLoader:
    """
    Anthropic-style skills loader
    Skills are folders with SKILL.md and instructions.md
    """
    
    def __init__(self, skills_base_path: str = None):
        """
        Initialize SkillLoader with optional custom path
        """
        if skills_base_path is None:
            # Default to 'skills' folder in project root
            self.skills_base_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "skills"
            )
        else:
            self.skills_base_path = skills_base_path
        
        print(f"📁 Skills path: {self.skills_base_path}")
        self.skills_cache = {}
        self._load_all_skills()
    
    def _load_all_skills(self):
        """Load all skills from skills directory"""
        if not os.path.exists(self.skills_base_path):
            os.makedirs(self.skills_base_path, exist_ok=True)
            self._create_default_skills()
        
        # Find all SKILL.md files
        skill_files = glob.glob(f"{self.skills_base_path}/**/SKILL.md", recursive=True)
        
        for skill_file in skill_files:
            skill_dir = os.path.dirname(skill_file)
            skill_name = os.path.basename(skill_dir)
            
            # Load SKILL.md
            with open(skill_file, 'r', encoding='utf-8') as f:
                skill_content = f.read()
            
            # Parse YAML frontmatter if exists
            skill_metadata = self._parse_skill_markdown(skill_content)
            
            # ========== FIX: Convert datetime objects to strings ==========
            if skill_metadata:
                for key, value in skill_metadata.items():
                    if isinstance(value, datetime):
                        skill_metadata[key] = value.isoformat()
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, datetime):
                                value[subkey] = subvalue.isoformat()
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, datetime):
                                value[i] = item.isoformat()
            
            # Load instructions.md if exists
            instructions_path = os.path.join(skill_dir, "instructions.md")
            instructions = ""
            if os.path.exists(instructions_path):
                with open(instructions_path, 'r', encoding='utf-8') as f:
                    instructions = f.read()
            
            # Load examples if exists
            examples = []
            examples_dir = os.path.join(skill_dir, "examples")
            if os.path.exists(examples_dir):
                for example_file in glob.glob(f"{examples_dir}/*.md"):
                    with open(example_file, 'r', encoding='utf-8') as f:
                        examples.append({
                            'name': os.path.basename(example_file),
                            'content': f.read()
                        })
            
            self.skills_cache[skill_name] = {
                'name': skill_name,
                'path': skill_dir,
                'metadata': skill_metadata,
                'instructions': instructions,
                'examples': examples
            }
        
        print(f"✅ Loaded {len(self.skills_cache)} skills")
    
    def _parse_skill_markdown(self, content: str) -> Dict:
        """Parse YAML frontmatter from markdown"""
        metadata = {}
        if content.startswith('---'):
            try:
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    import yaml
                    metadata = yaml.safe_load(parts[1])
                    # Ensure metadata is a dict
                    if not isinstance(metadata, dict):
                        metadata = {}
            except Exception as e:
                print(f"⚠️ Error parsing YAML: {e}")
                metadata = {}
        return metadata
    
    def _create_default_skills(self):
        """Create default skills if none exist"""
        from datetime import datetime
        
        default_skills = {
            'regression_skills': {
                'name': 'Regression Analysis',
                'description': 'Perform regression analysis on data',
                'instructions': """
You are a regression analysis expert. Follow these steps:
1. Check data quality and missing values
2. Select appropriate regression algorithm
3. Train model with cross-validation
4. Evaluate using R², RMSE, MAE
5. Generate predictions and insights
"""
            },
            'classification_skills': {
                'name': 'Classification Analysis',
                'description': 'Perform classification on data',
                'instructions': """
You are a classification expert. Follow these steps:
1. Encode categorical variables
2. Handle class imbalance
3. Train multiple classifiers
4. Evaluate using accuracy, precision, recall, F1
5. Generate confusion matrix and insights
"""
            }
        }
        
        for skill_dir, skill_info in default_skills.items():
            skill_path = os.path.join(self.skills_base_path, skill_dir)
            os.makedirs(skill_path, exist_ok=True)
            
            # Use string date instead of datetime object
            created_date = datetime.now().isoformat()
            
            # Create SKILL.md
            skill_md = f"""---
name: {skill_info['name']}
type: specialized
version: 1.0.0
author: System
created: {created_date}
---

# {skill_info['name']}

## Description
{skill_info['description']}

## Capabilities
- Data preprocessing
- Model training
- Evaluation metrics
- Visualization
- Insight generation

## Compatible With
- Claude API
- Custom Agents
"""
            with open(os.path.join(skill_path, "SKILL.md"), 'w', encoding='utf-8') as f:
                f.write(skill_md)
            
            # Create instructions.md
            with open(os.path.join(skill_path, "instructions.md"), 'w', encoding='utf-8') as f:
                f.write(skill_info['instructions'])
    
    def get_skill(self, skill_name: str) -> Optional[Dict]:
        """Get skill by name"""
        return self.skills_cache.get(skill_name)
    
    def find_skill_for_task(self, task_type: str, prompt: str = "") -> List[Dict]:
        """Find relevant skills based on task type"""
        relevant_skills = []
        
        for skill_name, skill in self.skills_cache.items():
            metadata = skill.get('metadata', {})
            
            # Match by task type
            if task_type in skill_name.lower():
                relevant_skills.append(skill)
            elif 'regression' in task_type and 'regression' in skill_name.lower():
                relevant_skills.append(skill)
            elif 'classification' in task_type and 'classification' in skill_name.lower():
                relevant_skills.append(skill)
            elif 'cluster' in task_type and 'cluster' in skill_name.lower():
                relevant_skills.append(skill)
        
        return relevant_skills
    
    def apply_skill_instructions(self, skill_name: str, context: Dict) -> str:
        """Apply skill instructions to current context"""
        skill = self.get_skill(skill_name)
        if not skill:
            return ""
        
        instructions = skill.get('instructions', '')
        
        # Replace placeholders with actual values
        for key, value in context.items():
            placeholder = f"{{{{ {key} }}}}"
            instructions = instructions.replace(placeholder, str(value))
        
        return instructions
    
    def list_all_skills(self) -> List[Dict]:
        """List all available skills with JSON-serializable data"""
        skills_list = []
        
        for name, data in self.skills_cache.items():
            metadata = data.get('metadata', {})
            
            # ========== FIX: Ensure all metadata is JSON serializable ==========
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    serializable_metadata[key] = value
                elif isinstance(value, (list, tuple)):
                    serializable_metadata[key] = [str(item) for item in value]
                elif isinstance(value, dict):
                    serializable_metadata[key] = {k: str(v) for k, v in value.items()}
                else:
                    serializable_metadata[key] = str(value)
            
            skills_list.append({
                'name': name,
                'path': str(data['path']),
                'metadata': serializable_metadata,
                'has_instructions': bool(data.get('instructions'))
            })
        
        return skills_list