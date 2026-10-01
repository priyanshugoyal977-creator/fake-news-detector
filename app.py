import os

import streamlit as st
import joblib

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Load trained ML model
model = joblib.load("model/fake_news_model.pkl")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fake News AI",
    page_icon="📰",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #070b14;
        color: white;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 56px;
        font-weight: 800;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 5px;
        letter-spacing: -2px;
    }

    .gradient-text {
        color: #818cf8;
    }

    .status-text {
        text-align: center;
        color: #34d399;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .main-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 40px;
    }

    .footer-text {
        text-align: center;
        color: #475569;
        font-size: 12px;
        margin-top: 50px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <h1 class="main-title">
        📰 FAKE <span class="gradient-text">NEWS AI</span>
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="status-text">
        ● AI NEWS INTELLIGENCE SYSTEM
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
        Investigate before you believe.<br>
        Compare your claim with current web evidence
        using Tavily, LangChain and Groq.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PIPELINE
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        🌐 **01 — CURRENT SEARCH**

        Tavily finds relevant current news.
        """
    )

with col2:
    st.info(
        """
        🧠 **02 — SEMANTIC CHECK**

        Groq compares the claim with the evidence.
        """
    )

with col3:
    st.info(
        """
        🤖 **03 — ML SIGNAL**

        Historical classifier provides an additional signal.
        """
    )

st.write("")


# =========================================================
# USER INPUT
# =========================================================

st.subheader("🔎 Start an Investigation")

st.caption(
    "Enter a news article, headline or claim. "
    "The system searches current evidence and then uses "
    "an AI model to compare the claim with that evidence."
)

news = st.text_area(
    "News Claim",
    placeholder=(
        "Example:\n"
        "India has announced a new national AI initiative..."
    ),
    height=200,
    label_visibility="collapsed"
)

investigate = st.button(
    "🔍 INVESTIGATE WITH AI",
    use_container_width=True
)


# =========================================================
# MAIN INVESTIGATION
# =========================================================

if investigate:

    if not news.strip():

        st.warning(
            "⚠️ Please enter a news claim or article first."
        )

        st.stop()


    # =====================================================
    # STEP 1 — CURRENT TAVILY SEARCH
    # =====================================================

    st.header("01 — Current Web Investigation")

    st.caption(
        "Searching current news sources related to your claim..."
    )

    articles = []
    source_text = ""


    # -----------------------------------------------------
    # CHECK TAVILY KEY
    # -----------------------------------------------------

    if not TAVILY_API_KEY:

        st.error(
            "❌ TAVILY_API_KEY is missing from your .env file."
        )

    else:

        with st.spinner(
            "🌐 Searching current news..."
        ):

            try:

                search = TavilySearch(
                    max_results=5,
                    topic="news",
                    time_range="week",
                    search_depth="basic"
                )

                results = search.invoke(news)

                articles = results.get(
                    "results",
                    []
                )

            except Exception as e:

                st.error(
                    "❌ Tavily search failed."
                )

                with st.expander(
                    "Show technical error"
                ):
                    st.code(str(e))


    # =====================================================
    # DISPLAY CURRENT SOURCES
    # =====================================================

    if articles:

        st.success(
            f"✓ {len(articles)} current sources found"
        )

        for i, article in enumerate(
            articles,
            1
        ):

            title = article.get(
                "title",
                "Untitled source"
            )

            url = article.get(
                "url",
                ""
            )

            content = article.get(
                "content",
                ""
            )

            # ---------------------------------------------
            # SOURCE CARD
            # ---------------------------------------------

            with st.container(
                border=True
            ):

                st.caption(
                    f"SOURCE {i}"
                )

                st.markdown(
                    f"**{title}**"
                )

                if url:

                    st.link_button(
                        "🔗 Open Source",
                        url
                    )

            # ---------------------------------------------
            # SOURCE TEXT FOR GROQ
            # ---------------------------------------------

            source_text += f"""
SOURCE {i}

TITLE:
{title}

URL:
{url}

CONTENT:
{content[:1800]}

--------------------------------
"""

    else:

        st.warning(
            "⚠️ No relevant current news sources were found."
        )


    # =====================================================
    # STEP 2 — GROQ SEMANTIC VERIFICATION
    # =====================================================

    st.header(
        "02 — AI Semantic Verification"
    )

    st.caption(
        "Groq compares the meaning of your claim with "
        "the retrieved current articles."
    )

    groq_result = None


    # -----------------------------------------------------
    # CHECK GROQ KEY
    # -----------------------------------------------------

    if not GROQ_API_KEY:

        st.error(
            "❌ GROQ_API_KEY is missing from your .env file."
        )

    elif not articles:

        st.warning(
            "AI verification cannot run because "
            "no current sources were retrieved."
        )

    else:

        with st.spinner(
            "🧠 AI is comparing the claim with current evidence..."
        ):

            try:

                llm = ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=0,
                    api_key=GROQ_API_KEY
                )


                # -----------------------------------------
                # GROQ PROMPT
                # -----------------------------------------

                prompt = f"""
You are the evidence verification engine of a
Fake News Detection application.

Your task is NOT to judge writing style.

Your task is to compare the MEANING and factual claims
in the USER CLAIM against the CURRENT NEWS SOURCES.

USER CLAIM:
{news}

CURRENT NEWS SOURCES:
{source_text}

Follow these steps:

STEP 1:
Identify the main factual claim made by the user.

STEP 2:
Read and compare the retrieved sources.

STEP 3:
Determine whether the sources provide evidence that:

- SUPPORTS the user's claim
- CONTRADICTS the user's claim
- Or does not provide enough evidence

STEP 4:
Consider dates carefully.

A current event should not be marked false simply
because the historical ML dataset does not contain it.

STEP 5:
Do not use the ML model's prediction as evidence.

STEP 6:
Do not invent information that is not present
in the retrieved sources.

IMPORTANT:

If the sources clearly report the same event or claim,
the result can be SUPPORTED.

If the sources clearly report information that conflicts
with the user's claim, the result can be CONTRADICTED.

If sources conflict, are weak, unrelated, or do not contain
enough information, use UNCERTAIN.

Return EXACTLY this format:

VERDICT:
SUPPORTED / CONTRADICTED / UNCERTAIN

CLAIM:
One sentence describing the user's main claim.

EVIDENCE:
Explain in 3-5 simple sentences how the current
sources compare with the claim.

SOURCE NUMBERS:
List the relevant source numbers.

CONFIDENCE:
HIGH / MEDIUM / LOW

LIMITATION:
Explain what the system cannot establish with certainty.
"""


                # -----------------------------------------
                # CALL GROQ
                # -----------------------------------------

                response = llm.invoke(
                    prompt
                )

                groq_result = response.content


                # -----------------------------------------
                # DISPLAY RESULT
                # -----------------------------------------

                st.subheader(
                    "🧠 AI Evidence Result"
                )

                st.info(
                    groq_result
                )


            except Exception as e:

                error_message = str(e)


                # -----------------------------------------
                # RATE LIMIT
                # -----------------------------------------

                if (
                    "429" in error_message
                    or
                    "rate_limit" in error_message.lower()
                    or
                    "quota" in error_message.lower()
                ):

                    st.warning(
                        "⚠️ Groq API rate limit or quota was reached."
                    )

                    st.caption(
                        "Tavily search worked, but the AI semantic "
                        "comparison could not be completed."
                    )


                # -----------------------------------------
                # AUTH ERROR
                # -----------------------------------------

                elif (
                    "401" in error_message
                    or
                    "authentication" in error_message.lower()
                    or
                    "invalid_api_key" in error_message.lower()
                ):

                    st.error(
                        "❌ Groq API key is invalid or unavailable."
                    )


                # -----------------------------------------
                # MODEL ERROR
                # -----------------------------------------

                elif (
                    "404" in error_message
                    or
                    "model_not_found" in error_message.lower()
                ):

                    st.error(
                        "❌ The selected Groq model is unavailable."
                    )

                    st.caption(
                        "Check the model name in the ChatGroq configuration."
                    )


                # -----------------------------------------
                # OTHER ERROR
                # -----------------------------------------

                else:

                    st.error(
                        "❌ AI semantic analysis failed."
                    )

                    with st.expander(
                        "Show technical error"
                    ):

                        st.code(
                            error_message
                        )


    # =====================================================
    # STEP 3 — ML HISTORICAL SIGNAL
    # =====================================================

    st.header(
        "03 — ML Historical Signal"
    )

    st.caption(
        "This model checks whether the writing resembles "
        "patterns from the historical training dataset. "
        "It does NOT verify whether the current event is "
        "true or false."
    )

    ml_signal = "UNAVAILABLE"

    ml_fake_probability = 0.0

    ml_real_probability = 0.0


    try:

        # ---------------------------------------------
        # GET PROBABILITIES
        # ---------------------------------------------

        probabilities = model.predict_proba(
            [news]
        )[0]


        # ---------------------------------------------
        # CLASSES
        #
        # 0 = Fake
        # 1 = Real
        # ---------------------------------------------

        ml_fake_probability = (
            probabilities[0] * 100
        )

        ml_real_probability = (
            probabilities[1] * 100
        )


        # ---------------------------------------------
        # HISTORICAL PATTERN SIGNAL
        # ---------------------------------------------

        if (
            ml_fake_probability
            >
            ml_real_probability
        ):

            ml_signal = (
                "MORE SIMILAR TO HISTORICAL FAKE NEWS"
            )

        else:

            ml_signal = (
                "MORE SIMILAR TO HISTORICAL REAL NEWS"
            )


        # ---------------------------------------------
        # DISPLAY ML SIGNAL
        # ---------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Historical Pattern Signal",
                ml_signal
            )

            st.metric(
                "Fake-pattern probability",
                f"{ml_fake_probability:.2f}%"
            )

            st.metric(
                "Real-pattern probability",
                f"{ml_real_probability:.2f}%"
            )


        with col2:

            st.info(
                """
                **Why this is NOT a fact-check**

                This model was trained on historical
                news articles.

                It looks at textual patterns learned
                from that dataset.

                A current event may look different
                from the historical training data.

                Therefore, this signal is NOT used
                to decide whether current news is
                true or false.
                """
            )


    except Exception as e:

        st.error(
            "ML historical signal could not be generated."
        )

        with st.expander(
            "Show technical error"
        ):

            st.code(
                str(e)
            )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    st.divider()

    st.header(
        "🎯 Investigation Result"
    )

    st.caption(
        "The final interpretation prioritizes current "
        "retrieved evidence and AI semantic comparison. "
        "The historical ML model is not treated as a "
        "fact-checker."
    )


    # =====================================================
    # EXTRACT GROQ VERDICT
    # =====================================================

    ai_verdict = "UNAVAILABLE"


    if groq_result:

        upper_result = groq_result.upper()


        if "VERDICT:" in upper_result:

            verdict_text = upper_result.split(
                "VERDICT:",
                1
            )[1].strip()


            if verdict_text.startswith(
                "SUPPORTED"
            ):

                ai_verdict = "SUPPORTED"


            elif verdict_text.startswith(
                "CONTRADICTED"
            ):

                ai_verdict = "CONTRADICTED"


            elif verdict_text.startswith(
                "UNCERTAIN"
            ):

                ai_verdict = "UNCERTAIN"


    # =====================================================
    # FINAL VERDICT DISPLAY
    # =====================================================

    if ai_verdict == "SUPPORTED":

        st.success(
            "🟢 CURRENT EVIDENCE: SUPPORTED"
        )

        st.write(
            "The retrieved current sources provide "
            "information consistent with the submitted claim."
        )

        st.caption(
            "This indicates that the available retrieved "
            "evidence supports the claim; it is not independent "
            "proof of truth."
        )


    elif ai_verdict == "CONTRADICTED":

        st.error(
            "🔴 CURRENT EVIDENCE: CONTRADICTED"
        )

        st.write(
            "The retrieved current sources contain information "
            "that conflicts with the submitted claim."
        )

        st.caption(
            "Review the cited sources before drawing a conclusion."
        )


    elif ai_verdict == "UNCERTAIN":

        st.warning(
            "🟡 CURRENT EVIDENCE: UNCERTAIN"
        )

        st.write(
            "The retrieved evidence does not provide enough "
            "consistent information to confidently support "
            "or contradict the claim."
        )


    else:

        st.info(
            "⚪ CURRENT EVIDENCE: UNAVAILABLE"
        )

        st.write(
            "The AI semantic comparison could not be completed."
        )

        st.caption(
            "The historical ML result should NOT be used "
            "as a substitute for current verification."
        )


    # =====================================================
    # SIGNAL COMPARISON
    # =====================================================

    st.subheader(
        "📊 Signal Comparison"
    )

    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # HISTORICAL ML
    # -----------------------------------------------------

    with col1:

        st.info(
            f"""
            **🤖 HISTORICAL ML SIGNAL**

            {ml_signal}

            Fake-pattern probability:
            **{ml_fake_probability:.2f}%**

            Real-pattern probability:
            **{ml_real_probability:.2f}%**

            Based only on historical text patterns.
            """
        )


    # -----------------------------------------------------
    # CURRENT WEB
    # -----------------------------------------------------

    with col2:

        st.info(
            f"""
            **🌐 CURRENT WEB EVIDENCE**

            **{len(articles)} SOURCES**

            Current articles retrieved by Tavily
            for evidence comparison.
            """
        )


    # -----------------------------------------------------
    # AI VERDICT
    # -----------------------------------------------------

    with col3:

        if ai_verdict == "SUPPORTED":

            st.success(
                """
                **🧠 AI VERDICT**

                **SUPPORTED**

                Current retrieved evidence supports
                the submitted claim.
                """
            )

        elif ai_verdict == "CONTRADICTED":

            st.error(
                """
                **🧠 AI VERDICT**

                **CONTRADICTED**

                Current retrieved evidence conflicts
                with the submitted claim.
                """
            )

        elif ai_verdict == "UNCERTAIN":

            st.warning(
                """
                **🧠 AI VERDICT**

                **UNCERTAIN**

                The available evidence is not sufficient
                to support or contradict the claim.
                """
            )

        else:

            st.info(
                """
                **🧠 AI VERDICT**

                **UNAVAILABLE**

                AI evidence comparison could not
                be completed.
                """
            )


    # =====================================================
    # IMPORTANT EXPLANATION
    # =====================================================

    st.subheader(
        "🔎 How to Interpret the Results"
    )

    st.info(
        """
        **Historical ML Signal**

        Shows whether the writing resembles patterns
        learned from the historical fake/real dataset.


        **Current Web Evidence**

        Shows how many relevant current articles
        were retrieved by Tavily.


        **AI Verdict**

        Groq compares the meaning of the submitted
        claim against the retrieved current evidence.


        **Important**

        The historical ML model is NOT used as an
        independent fact-checker.

        The ML prediction should therefore not be
        interpreted as proof that a current article
        is real or fake.
        """
    )


# =========================================================
# TECHNOLOGIES
# =========================================================

st.divider()

st.header(
    "🧩 Built With"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.info(
        """
        🐍 **Python**

        Core programming language.
        """
    )


with col2:

    st.info(
        """
        🤖 **Machine Learning**

        TF-IDF + Logistic Regression.
        """
    )


with col3:

    st.info(
        """
        🌐 **Tavily + LangChain**

        Current web retrieval.
        """
    )


with col4:

    st.info(
        """
        🧠 **Groq AI**

        Semantic evidence analysis.
        """
    )


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander(
    "ℹ️ How Fake News AI Works"
):

    st.markdown(
        """
        ### Step 1 — Current Search

        Tavily searches current news sources related
        to the submitted claim.


        ### Step 2 — Evidence Collection

        Relevant article titles, URLs and content
        are collected.


        ### Step 3 — Semantic Comparison

        Groq receives the user's claim and the
        retrieved articles.

        It compares their meaning rather than
        simply matching individual words.


        ### Step 4 — Evidence Verdict

        The AI returns:

        - SUPPORTED
        - CONTRADICTED
        - UNCERTAIN


        ### Step 5 — Historical ML Signal

        The original TF-IDF + Logistic Regression
        model is used only as an additional
        historical text-pattern signal.


        ### Important Limitation

        The ML model does not know whether a
        current event actually happened.

        Current claims should therefore be checked
        against the original sources.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "FAKE NEWS AI · Current News Evidence Analysis"
)

st.caption(
    "Tavily · LangChain · Groq · Machine Learning"
)

st.caption(
    "⚠️ AI analysis is an assistive signal, "
    "not independent proof of factual truth."
)