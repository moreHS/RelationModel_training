import json
import re
import os
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Any, Tuple, Set

class KnowledgeParser:
    """
    설명: sLLM이 추론을 통해 뱉어낸 Raw Text 결과를 파싱하여 평가 Metric (Exact Match - F1 Score)을 계산 할 수 있도록 후처리 하는 클래스 입니다. 
    """
    def __init__(self, prompt_config_path: str = None):
        """
        Input: 프롬프트 설정 YAML 파일 경로
        Output: None
        설명: 파서 초기화 및 Task 별 Relation 설정. YAML 파일이 주어지면 프롬프트 지문 (Signature)을 로드 하여 dyanmic_signatures에 저장함 (Task를  구분 할 수 있는 프롬프트 내용)
        """
        self.default_relations = {
            "NER-BEE": "true",
            "NER-BEE_TRUE_ONLY": "true",
            "NER-NER": "no_relation",
            "BEE-BEE": "no_relation",
            "COMBINE_ALL": "true",
            "UNKNOWN": "true"
        }
        
        self.dynamic_signatures = {}
        if prompt_config_path and os.path.exists(prompt_config_path):
            self._load_signatures_from_file(prompt_config_path)

    def _load_signatures_from_file(self, path: str):
        """
        Input: Prompt YAML 파일 경로
        Output: None
        설명: YAML 설정 파일에서 Task별 System Prompt를 읽어와 첫 250자로 고유 지문을 생성.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            task_map = {
                "system_prompt_combine_all": "COMBINE_ALL",
                "system_prompt_ner_bee_true_only": "NER-BEE_TRUE_ONLY",
                "system_prompt_ner_bee": "NER-BEE",
                "system_prompt_ner_ner": "NER-NER",
                "system_prompt_bee_bee": "BEE-BEE",
            }

            for config_key, task_name in task_map.items():
                pattern = rf"{config_key}\s*:\s*\|(.*?)(?=\n[a-zA-Z0-9_]+\s*:\s*\||\Z)"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    raw_prompt = match.group(1).strip()
                    # 1. Clean out comments (just like the vLLM script does)
                    clean_prompt = re.sub(r'\n\s*#.*', '', '\n' + raw_prompt)
                    # 2. Squash whitespace
                    fingerprint = re.sub(r'\s+', '', clean_prompt)
                    # 3. Store only the first 250 characters as the unique signature
                    self.dynamic_signatures[task_name] = fingerprint[:250]
                    
            print(f"✅ Loaded {len(self.dynamic_signatures)} dynamic prompt signatures from config.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load prompt config. Using fallbacks. ({e})")

    def determine_task_from_prompt(self, prompt: str) -> str:
        """
        Input: prompt text
        Ouptput: task_name - 판별된 태스트 이름 (NERNER, NERBEE, BEEBEE etc.)
        설명: 입력된 프롬프트의 공백을 제거한 뒤, 지문(signature)과 대조.
        """
        if not prompt: 
            return "COMBINE_ALL" # 🎯 Default fallback
            
        normalized_prompt = re.sub(r'\s+', '', prompt)
        
        # 1. Check Dynamic YAML Signatures first
        if self.dynamic_signatures:
            for task_name, signature in sorted(self.dynamic_signatures.items(), key=lambda x: len(x[1]), reverse=True):
                if signature in normalized_prompt:
                    return task_name

        # 2. Hardcoded Fallbacks
        if "Therearethreetasksandthreerespectivedescriptions:" in normalized_prompt: return "COMBINE_ALL"
        if "Printonly[true]inthe\"is_relational\"keyifarelationisdetected" in normalized_prompt: return "NER-BEE_TRUE_ONLY"
        if "performabinaryclassificationofwhethertheentitypairhasarelation" in normalized_prompt: return "NER-BEE"
        if "classifytheentitypairintooneoftherelationsinthe<NER-NERRelationList>" in normalized_prompt: return "NER-NER"
        if "classifytheentitypairin<CANDIDATE_PAIRS>intooneoftherelationsinthe<BEE-BEERelationList>" in normalized_prompt: return "BEE-BEE"
            
        # 3. 🎯 AGGRESSIVE KEYWORD FALLBACK (Kills "UNKNOWN")
        upper_prompt = prompt.upper()
        if "COMBINE_ALL" in upper_prompt or "THREE TASKS" in upper_prompt:
            return "COMBINE_ALL"
        if "NER-BEE_TRUE_ONLY" in upper_prompt:
            return "NER-BEE_TRUE_ONLY"
        if "NER-BEE" in upper_prompt or "IS_RELATIONAL" in upper_prompt:
            return "NER-BEE"
        if "BEE-BEE" in upper_prompt:
            return "BEE-BEE"
        if "NER-NER" in upper_prompt:
            return "NER-NER"

        # 4. Final safety net
        return "COMBINE_ALL"

    @staticmethod
    def clean_json_string(val: str) -> Any:
        """
        Input: LLM이 생성한 RAW Output
        Output: Parsing 완료된 JSON 객체
        설명: LLM 텍스트 특유의 노이즈 regex로 제거(Markdown tick, Response Prefix -- '''json''')
        """
        if not val or not isinstance(val, str): return val
        
        val = val.replace("### Response:", "")
        val = re.sub(r"```json\n?|```", "", val).strip()
        
        try:
            return json.loads(val)
        except:
            return val

    def parse_inference(self, prompt: str, raw_response: str) -> List[list]:
        """
        Input: prompt, LLM의 Raw Output
        Output: List[List] 형태의 [[pair_id, subject, object, relation]...]
        설명: 프롬프트를 분석하여 Task를 찾아내고, 응답을 파싱하여 scikit-learn 등의 평가 모듈에 바로 사용할 수 있도록 2차원 List를 반환하는 함수
        """
        expected_task = self.determine_task_from_prompt(prompt)
        triplets = self.parse(raw_response, expected_task)
        return [list(t) for t in triplets]

    def parse(self, data: Any, expected_task: str = "COMBINE_ALL") -> Set[Tuple[str, str, str, str]]:
        """
        Input: 정제된 JSON 데이터 또는 string, 예상 Task name
        Output: 순서와 중복을 무시할 수 있는 Set(Tuple)
        설명: 파싱된 데이터 구조 (Dict, List)를 순회하여 (pair_id, subject, object, relation) Triplet을 추출함. 문자열 형식인 경우 정규식 백업 파서로 넘김
        """
        data = self.clean_json_string(data)

        if isinstance(data, str):
            return self._parse_via_regex(data, expected_task)

        triplets = set()
        
        if isinstance(data, dict):
            found_hierarchical = False
            # 🎯 Added ALL 5 possible task keys here!
            for key in ["NER-NER", "NER-BEE", "NER-BEE_TRUE_ONLY", "BEE-BEE", "COMBINE_ALL"]:
                if key in data:
                    found_hierarchical = True
                    items = data[key] if isinstance(data[key], list) else []
                    for item in items:
                        t = self._canonicalize(item, key)
                        if t: triplets.add(t) 
            
            if not found_hierarchical and "subject" in data and "object" in data:
                t = self._canonicalize(data, expected_task)
                if t: triplets.add(t)

        elif isinstance(data, list):
            for item in data:
                t = self._canonicalize(item, expected_task)
                if t: triplets.add(t)

        return triplets

    def _extract_word(self, val: Any) -> str:
        """
        Input: Dict 형태의 Entity 정보 또는 문자열
        Output: 정제된 단어(word) 문자열
        설명: 입력값이 딕셔너리 포맷 ({"word": "사과"})일 경우 "사과"만 안전하게 추출하고, LLM이 생성한 불필요한 따옴표/백슬래시 노이즈를 제거함.
        """
        if isinstance(val, dict):
            word = str(val.get("word", ""))
        else:
            word = str(val)
            
        # Strip rogue quotation marks and escape slashes injected by the LLM
        return word.strip().replace('"', '').replace('\\', '')

    def _canonicalize(self, item: Dict, task_type: str) -> Tuple[str, str, str, str]:
        """
        Input: 추출된 단일 Relation 딕셔너리 데이터
        Output: (pair_id, subject, object, relation) Lower-case Triplet
        설명: 대소문자 불일치로 인한 False Negative 처리를 막기 위해 모든 문자열을 소문자로 치환. NER-BEE의 괄호 태그 및 잘못된 Relation 명칭 보정 기능 포함.
        """
        if not isinstance(item, dict): return None
        
        actual_task = task_type
        
        # 🎯 1. Grab pair_id
        pair_id = str(item.get("pair_id", "UNKNOWN")).strip()
            
        sub = self._extract_word(item.get("subject", "")).lower()
        obj = self._extract_word(item.get("object", "")).lower()
        
        rel = item.get("relation")
        is_rel = item.get("is_relational")
        
        if rel is None:
            final_rel = str(is_rel).lower() if is_rel is not None else self.default_relations.get(actual_task, "true")
        else:
            final_rel = str(rel).lower()
            
        if actual_task in ["NER-BEE", "NER-BEE_TRUE_ONLY"]:
            sub = re.sub(r'\s*\([^)]*\)$', '', sub).strip()
            obj = re.sub(r'\s*\([^)]*\)$', '', obj).strip()
            
            clean_rel = final_rel.strip().replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            if clean_rel not in ["false", "no_relation", "0"]:
                clean_rel = "true"
            final_rel = clean_rel
        else:
            final_rel = final_rel.strip().replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            
        if sub and obj:
            # 🎯 Return 4-tuple now
            return (pair_id, sub, obj, final_rel)
        return None

    def _parse_via_regex(self, text: str, expected_task: str) -> Set[Tuple[str, str, str, str]]:
        """
        Input: JSON 문법이 깨진 LLM RAW 텍스트
        Output: 정규식으로 스크래핑된 (pair_id, subject, object, relation) Tuples
        설명: LLM이 "," "()"를 누락하여 JSON 파싱이 불가능할 때 호출 되는 백업 로직. 정규식을 사용해 텍스트 내에서 패턴을 강제로 긁어냄.
        """
        triplets = set()
        objects = re.findall(r'\{(.*?)\}', text, re.DOTALL)
        for obj_str in objects:
            # 🎯 2. Grab pair_id via regex
            pid_match = re.search(r'"pair_id"\s*:\s*"([^"]+)"', obj_str)
            sub = re.search(r'"subject"\s*:\s*"([^"]+)"', obj_str)
            obj = re.search(r'"object"\s*:\s*"([^"]+)"', obj_str)
            rel = re.search(r'"relation"\s*:\s*"([^"]+)"', obj_str)
            is_rel = re.search(r'"is_relational"\s*:\s*(true|false|[^,\s\}]+)', obj_str)
            
            p_id = pid_match.group(1) if pid_match else "UNKNOWN"
            s = sub.group(1) if sub else None
            o = obj.group(1) if obj else None
            r = "true"
            
            if rel: r = rel.group(1)
            elif is_rel: r = is_rel.group(1)
            
            if s and o:
                clean_s = s.replace('\\"', '"').replace("\\'", "'").strip().lower()
                clean_o = o.replace('\\"', '"').replace("\\'", "'").strip().lower()
                clean_r = str(r).strip().lower().replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                
                if expected_task in ["NER-BEE", "NER-BEE_TRUE_ONLY"]:
                    clean_s = re.sub(r'\s*\([^)]*\)$', '', clean_s).strip()
                    clean_o = re.sub(r'\s*\([^)]*\)$', '', clean_o).strip()
                    if clean_r not in ["false", "no_relation", "0"]:
                        clean_r = "true"
                        
                # 🎯 Return 4-tuple
                triplets.add((p_id, clean_s, clean_o, clean_r))
        return triplets


if __name__ == "__main__":
    """
    설명: 파서 디버깅 및 Audit을 위한 메인 블록. vLLM 예측 결과 파일에서 파싱이 정상적으로 이루어지는지 전체를 파일로 저장하고, 처음 5개만 콘솔에 출력함. 
    """
    # =========================================================
    # ⚙️ CONFIGURATION & TOGGLES
    # =========================================================
    TARGET_MODEL = "qwen" 
    MODE = "new_prompt" 
    
    # 🎯 THE TOGGLE BUTTON
    EVAL_MODE = True  
    
    INPUT_FILE = f"vllm_predictions_{TARGET_MODEL}_{MODE}.jsonl"
    OUTPUT_FILE = f"parsed_output_{TARGET_MODEL}_{MODE}.jsonl"
    PROMPT_CONFIG_FILE = "sft_data_generation_prompts_edited.yaml" 
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
    else:
        parser = KnowledgeParser(prompt_config_path=PROMPT_CONFIG_FILE)

        # =========================================================
        # 🔄 PARSE vLLM OUTPUTS FOR EVALUATION
        # =========================================================
        print(f"🔄 Parsing predictions from {INPUT_FILE}...")
        
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            
            for i, line in enumerate(tqdm(f_in, desc="Parsing lines")):
                data = json.loads(line)
                raw_pred = data.get('prediction', "")
                full_input = data.get('full_input', "")

                # 1. Detect task directly from the prompt (since this is the vLLM output file)
                expected_task = parser.determine_task_from_prompt(full_input)

                # 2. Parse the tuples (Returns: pair_id, sub, obj, rel)
                pred_tuples = parser.parse(raw_pred, expected_task=expected_task)

                # 3. Build the final parsed record for Evaluation
                parsed_record = {
                    "expected_task": expected_task,
                    "pred_tuples": [list(t) for t in pred_tuples],
                    "full_input": full_input
                }
                
                # 4. IF IN EVAL MODE: Grab and parse the Ground Truth
                if EVAL_MODE:
                    raw_gold = data.get('ground_truth', "")
                    gold_tuples = parser.parse(raw_gold, expected_task=expected_task)
                    parsed_record["ground_truth"] = [list(t) for t in gold_tuples]
                
                # 🎯 Save to Output File
                f_out.write(json.dumps(parsed_record, ensure_ascii=False) + "\n")
                
                # 🎯 Print Audit
                if i < 5:
                    if i == 0: print("\n🔍 --- AUDIT OF FIRST 5 ROWS --- 🔍")
                    print(json.dumps(parsed_record, ensure_ascii=False, indent=2))
                    
        print(f"\n✅ Parsing complete! Ready for metric calculation. Saved to: {OUTPUT_FILE}")