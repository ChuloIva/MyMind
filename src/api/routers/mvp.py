# /src/6_api/routers/mvp.py

from fastapi import APIRouter, UploadFile, File, HTTPException
import logging

# Import the actual analysis functions from your modules
from src.preprocessing.llm_processing.keyword_extraction import KeywordExtractor
from src.analysis.therapeutic_methods.distortions import CognitiveDistortionAnalyzer
from src.preprocessing.llm_processing.needs_extraction import NeedsExtractor
from src.common.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["MVP Analysis"])

@router.post("/simple_analyze")
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
        keyword_extractor = KeywordExtractor()
        dummy_segments = [{'text': text, 'start': 0, 'end': len(text)/1000}]
        keywords_analysis = keyword_extractor.extract_keywords_and_sentiment(dummy_segments)
        logger.info("Keyword extraction complete.")

        # 2. Perform Therapeutic Analysis
        distortion_analyzer = CognitiveDistortionAnalyzer()
        dummy_session_data = {'processed_segments': [{'text': text, 'speaker': 'client', 'start': 0}]}
        therapeutic_analysis = distortion_analyzer.analyze_session(dummy_session_data)
        logger.info("Therapeutic analysis complete.")

        # 3. Perform Needs Analysis (if OpenAI API key is available)
        needs_analysis = None
        try:
            if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
                needs_extractor = NeedsExtractor(api_key=settings.openai_api_key)
                dummy_segments = [{'text': text, 'start': 0, 'end': len(text)/1000}]
                needs_analysis = needs_extractor.extract_needs_and_segments(dummy_segments)
                logger.info("Needs analysis complete.")
        except Exception as e:
            logger.warning(f"Needs analysis failed (likely missing OpenAI API key): {e}")

        # 4. Combine and return the results
        result = {
            "keywords_analysis": keywords_analysis,
            "therapeutic_analysis": therapeutic_analysis
        }
        
        if needs_analysis:
            result["needs_analysis"] = needs_analysis
            
        return result
    except Exception as e:
        logger.error(f"Core analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during AI analysis: {str(e)}")