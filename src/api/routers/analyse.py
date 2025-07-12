# /src/api/routers/analyse.py

from fastapi import APIRouter, UploadFile, File, HTTPException
import logging

# Import the actual analysis functions from your modules
from preprocessing.llm_processing.keyword_extraction import KeywordExtractor
from analysis.therapeutic_methods.distortions import CognitiveDistortionAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyse/{session_id}")
async def analyse_session(session_id: str):
    # This is your original placeholder endpoint. We can keep it or remove it.
    # For now, it's fine to leave it.
    return {"message": f"Analyzing session {session_id} (Full pipeline not implemented yet)"}

@router.post("/simple_analyze", tags=["MVP Analysis"])
async def simple_text_analysis(file: UploadFile = File(...)):
    """
    Accepts a .txt file, analyzes it for keywords and therapeutic patterns,
    and returns the combined results. This is a simplified analysis flow.
    """
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .txt file.")

    try:
        contents = await file.read()
        text = contents.decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty or contains only whitespace.")

    logger.info("Starting simple analysis for uploaded text file.")

    try:
        # 1. Perform Keyword Extraction
        # We instantiate the class from the module and call its method.
        # We must adapt our simple text input to the format the function expects.
        keyword_extractor = KeywordExtractor()
        # The function expects a list of segments, so we create a dummy one.
        dummy_segments = [{'text': text, 'start': 0, 'end': len(text)/1000}]
        keywords_analysis = keyword_extractor.extract_keywords_and_sentiment(dummy_segments)
        logger.info("Keyword extraction complete.")

        # 2. Perform Therapeutic Analysis
        # We instantiate the analyzer class and call its main method.
        distortion_analyzer = CognitiveDistortionAnalyzer()
        # This function expects session_data with segments, so we create a dummy structure.
        dummy_session_data = {
            'processed_segments': [{'text': text, 'speaker': 'client', 'start': 0}]
        }
        therapeutic_analysis = distortion_analyzer.analyze_session(dummy_session_data)
        logger.info("Therapeutic analysis complete.")

        # 3. Combine and return the results
        return {
            "keywords_analysis": keywords_analysis,
            "therapeutic_analysis": therapeutic_analysis
        }
    except Exception as e:
        logger.error(f"Core analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during AI analysis: {str(e)}")