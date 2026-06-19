from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "apply_generic_memory_ops", ROOT / "scripts" / "apply_generic_memory_ops.py"
)
assert SPEC is not None
ops = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ops
SPEC.loader.exec_module(ops)


def _chunk(text: str):
    return ops.MemoryChunk(
        question_id="q",
        node_type="leaf",
        session_id="s",
        session_date="2026/05/22",
        turn_index=0,
        text=text,
    )


def test_speaker_mismatch_abstains_when_fact_belongs_to_other_speaker() -> None:
    result = ops.op_speaker_mismatch_abstain(
        "What does Melanie's necklace symbolize?",
        [
            _chunk(
                "User (Caroline) -> Melanie: My necklace from grandma symbolizes "
                "love, faith, and strength."
            ),
            _chunk("Assistant (Melanie) -> Caroline: That necklace is beautiful."),
        ],
    )

    assert result is not None
    assert result.operator == "speaker_mismatch_abstain"
    assert "does not mention this information for Melanie" in result.answer


def test_speaker_mismatch_keeps_target_speaker_fact() -> None:
    result = ops.op_speaker_mismatch_abstain(
        "What does Melanie's necklace symbolize?",
        [
            _chunk(
                "Assistant (Melanie) -> Caroline: My necklace symbolizes "
                "love, faith, and strength."
            ),
        ],
    )

    assert result is None


def test_speaker_mismatch_keeps_later_sentence_from_target_speaker_line() -> None:
    result = ops.op_speaker_mismatch_abstain(
        "When did Melanie run a charity race?",
        [
            _chunk(
                "Assistant (Melanie) -> Caroline: Hey Caroline, lots happened. "
                "I ran a charity race for mental health last Saturday."
            ),
            _chunk("User (Caroline) -> Melanie: That charity race sounds great, Mel!"),
        ],
    )

    assert result is None


def test_speaker_mismatch_ignores_target_speaker_acknowledgement() -> None:
    result = ops.op_speaker_mismatch_abstain(
        "What did Caroline realize after her charity race?",
        [
            _chunk(
                "Assistant (Melanie) -> Caroline: I ran a charity race for mental health last "
                "Saturday. I'm starting to realize that self-care is really important."
            ),
            _chunk(
                "User (Caroline) -> Melanie: That charity race sounds great, Mel! "
                "I'm proud of you."
            ),
        ],
    )

    assert result is not None
    assert result.operator == "speaker_mismatch_abstain"


def test_generic_speaker_mismatch_strict_mode_only_for_abstention_category() -> None:
    chunks = [
        _chunk(
            "Assistant (Melanie) -> Caroline: I ran a charity race for mental health last "
            "Saturday. I'm starting to realize that self-care is really important."
        ),
        _chunk(
            "User (Caroline) -> Melanie: That charity race sounds great, Mel! "
            "I'm proud of you."
        ),
    ]

    normal = ops.generic_answer(
        "What did Caroline realize after her charity race?",
        "category_1",
        chunks,
        enable_speaker_mismatch_abstain=True,
    )
    abstention = ops.generic_answer(
        "What did Caroline realize after her charity race?",
        "category_5",
        chunks,
        enable_speaker_mismatch_abstain=True,
    )

    assert normal is None
    assert abstention is not None
    assert abstention.operator == "speaker_mismatch_abstain"


def test_binary_speaker_fact_answers_no_when_other_speaker_made_item() -> None:
    result = ops.generic_answer(
        "Did Caroline make the black and white bowl in the photo?",
        "category_5",
        [
            _chunk("User (Caroline) -> Melanie: That bowl is gorgeous. Did you make it?"),
            _chunk(
                "Assistant (Melanie) -> Caroline: Thanks, Caroline! Yeah, "
                "I made this bowl in my pottery class."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is not None
    assert result.operator == "binary_speaker_fact"
    assert result.answer == "No"
    assert "Melanie" in result.reason


def test_binary_speaker_fact_ignores_make_feel_and_make_sure() -> None:
    result = ops.generic_answer(
        "Did Caroline make the black and white bowl in the photo?",
        "category_5",
        [
            _chunk(
                "User (Caroline) -> Melanie: The support group has made me feel "
                "accepted. I'll make sure the kids have a safe home."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_binary_speaker_fact_ignores_adjectival_hand_painted_object() -> None:
    result = ops.generic_answer(
        "Did Caroline make the black and white bowl in the photo?",
        "category_5",
        [
            _chunk(
                "User (Caroline) -> Melanie: I've got some other sentimental "
                "stuff, like my hand-painted bowl."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_binary_speaker_fact_ignores_abstract_make_happen_difference() -> None:
    result = ops.generic_answer(
        "Did Caroline make the black and white bowl in the photo?",
        "category_5",
        [
            _chunk(
                "User (Caroline) -> Melanie: My own journey made a huge difference. "
                "I want to help make that happen for others."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_binary_speaker_fact_answers_yes_when_target_speaker_made_item() -> None:
    result = ops.generic_answer(
        "Did Melanie make the black and white bowl in the photo?",
        "category_5",
        [
            _chunk(
                "Assistant (Melanie) -> Caroline: Thanks, Caroline! Yeah, "
                "I made this bowl in my pottery class."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is not None
    assert result.operator == "binary_speaker_fact"
    assert result.answer == "Yes"


def test_binary_speaker_fact_answers_no_for_pet_owned_by_other_speaker() -> None:
    result = ops.generic_answer(
        "Is Oscar Melanie's pet?",
        "category_5",
        [
            _chunk(
                "User (Caroline) -> Melanie: Oscar is my guinea pig. "
                "He loves parsley and hay."
            ),
            _chunk("Assistant (Melanie) -> Caroline: Oliver is my dog."),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is not None
    assert result.operator == "binary_speaker_fact"
    assert result.answer == "No"
    assert "Caroline" in result.reason


def test_binary_speaker_fact_does_not_handle_general_cat_preference() -> None:
    result = ops.generic_answer(
        "Does Deborah like cats?",
        "category_5",
        [
            _chunk(
                "User (Deborah) -> Jolene: I don't like dogs, that's why I have cats."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_book_recommendations_extracts_target_speaker_quoted_book() -> None:
    result = ops.generic_answer(
        "What book did Caroline recommend to Melanie?",
        "category_4",
        [
            _chunk(
                'User (Caroline) -> Melanie: I loved "Becoming Nicole" by Amy Ellis Nutt. '
                "Highly recommend it for sure!"
            ),
            _chunk('Assistant (Melanie) -> Caroline: I recommend "Charlotte\'s Web" too.'),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is not None
    assert result.operator == "book_recommendations"
    assert result.answer == "Becoming Nicole"


def test_book_recommendations_handles_recommendations_given_form() -> None:
    result = ops.generic_answer(
        "What book recommendations has Joanna given to Nate?",
        "category_1",
        [
            _chunk('User (Joanna) -> Nate: You should read "Little Women" sometime.'),
            _chunk('User (Joanna) -> Nate: I think you would love "A Court of Thorns and Roses".'),
            _chunk('Assistant (Nate) -> Joanna: I recommend "Project Hail Mary".'),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is not None
    assert result.operator == "book_recommendations"
    assert result.answer == "A Court of Thorns and Roses"


def test_book_recommendations_ignores_non_book_recommendations() -> None:
    result = ops.generic_answer(
        "What book did Caroline recommend to Melanie?",
        "category_4",
        [
            _chunk('User (Caroline) -> Melanie: You would love the movie "Little Women".'),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_book_recommendations_ignores_took_your_movie_recommendation() -> None:
    result = ops.generic_answer(
        "What book recommendations has Joanna given to Nate?",
        "category_1",
        [
            _chunk(
                'User (Joanna) -> Nate: I took your reccomendation and watched '
                '"The Lord of the Rings" Trilogy last night!'
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_direct_favorite_ignores_non_food_top_pick() -> None:
    result = ops.generic_answer(
        "What is Jon's favorite style of dance?",
        "category_4",
        [
            _chunk(
                "User (Jon) -> Gina: I love all dances, but contemporary is my top pick. "
                "It's so expressive and powerful!"
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_direct_favorite_extracts_top_of_list_for_dish_question() -> None:
    result = ops.generic_answer(
        "What is Nate's favorite dish from the cooking show he hosted?",
        "category_4",
        [
            _chunk(
                "Assistant (Nate) -> Joanna: Coconut milk ice cream is at the top "
                "of my list. It's so smooth and creamy."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is not None
    assert result.operator == "direct_favorite"
    assert result.answer == "Coconut milk ice cream"


def test_direct_favorite_ignores_recipe_only_question() -> None:
    result = ops.generic_answer(
        "What is Audrey's favorite recipe that she shares with Andrew?",
        "category_4",
        [
            _chunk(
                "User (Audrey) -> Andrew: Roasted Chicken is one of my favorites - "
                "sure I'll send you the recipe in a bit."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_direct_favorite_ignores_broad_dessert_question() -> None:
    result = ops.generic_answer(
        "What are Nate's favorite desserts?",
        "category_1",
        [
            _chunk(
                "Assistant (Nate) -> Joanna: I love coconut milk, but I also enjoy "
                "chocolate and mixed berry flavors."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_direct_favorite_ignores_question_lines() -> None:
    result = ops.generic_answer(
        "What is Nate's favorite dish?",
        "category_4",
        [
            _chunk("User (Joanna) -> Nate: What's your favorite dish from the show?"),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert result is None


def test_direct_favorite_requires_matching_question_domain() -> None:
    book_result = ops.generic_answer(
        "What are Deborah's favorite books?",
        "category_5",
        [
            _chunk(
                "User (Deborah) -> Jolene: This is my favorite studio and it's "
                "always so calming."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )
    game_result = ops.generic_answer(
        "What is Nate's favorite video game?",
        "category_4",
        [
            _chunk(
                "Assistant (Nate) -> Joanna: Coconut milk ice cream is at the top "
                "of my list."
            ),
        ],
        enable_speaker_mismatch_abstain=True,
    )

    assert book_result is None
    assert game_result is None


def test_speaker_mismatch_keeps_target_speaker_opinion() -> None:
    result = ops.op_speaker_mismatch_abstain(
        "What does Melanie think about Caroline's decision to adopt?",
        [
            _chunk("User (Caroline) -> Melanie: I applied to adoption agencies."),
            _chunk("Assistant (Melanie) -> Caroline: Adoption sounds awesome. Wishing you the best."),
        ],
    )

    assert result is None


def test_sum_amounts_is_disabled_in_default_generic_chain() -> None:
    chunks = [
        _chunk("User: I bought a bike helmet for $120."),
        _chunk("User: I bought bike lights for $40."),
    ]

    direct = ops.op_sum_money(
        "How much total money have I spent on bike-related expenses?",
        chunks,
    )
    default = ops.generic_answer(
        "How much total money have I spent on bike-related expenses?",
        "multi-session",
        chunks,
    )
    enabled = ops.generic_answer(
        "How much total money have I spent on bike-related expenses?",
        "multi-session",
        chunks,
        enable_sum_amounts=True,
    )

    assert direct is not None
    assert default is None
    assert enabled is not None
    assert enabled.operator == "sum_amounts"


def test_sum_amounts_rejects_duration_spent_questions() -> None:
    result = ops.op_sum_money(
        "What is the total number of days I spent in Japan and Chicago?",
        [
            _chunk("User: I visited Japan from April 15 to April 22."),
            _chunk("Assistant: A hotel option could cost around $40."),
        ],
    )

    assert result is None


def test_sum_unit_quantities_adds_two_named_trip_durations() -> None:
    result = ops.op_sum_unit_quantities(
        "How many days did I spend in total traveling in Hawaii and in New York City?",
        [
            _chunk("User: I recently got back from an island-hopping trip to Hawaii for 10 days."),
            _chunk("User: I recently got back from a solo trip to New York City for five days."),
        ],
    )

    assert result is not None
    assert result.operator == "sum_unit_quantities"
    assert result.answer == "15 days"


def test_sum_unit_quantities_handles_week_and_half() -> None:
    result = ops.op_sum_unit_quantities(
        "How many weeks did it take me to watch all the Marvel Cinematic Universe movies and the main Star Wars films?",
        [
            _chunk("User: I watched all 22 Marvel Cinematic Universe movies in two weeks."),
            _chunk("User: I watched all the main Star Wars films in a week and a half."),
        ],
    )

    assert result is not None
    assert result.answer == "3.5 weeks"


def test_sum_unit_quantities_requires_every_anchor() -> None:
    result = ops.op_sum_unit_quantities(
        "How many days did I spend in total traveling in Hawaii and in Seattle?",
        [_chunk("User: I recently got back from an island-hopping trip to Hawaii for 10 days.")],
    )

    assert result is None


def test_count_health_visits_counts_distinct_doctor_types() -> None:
    result = ops.op_count_health_visits(
        "How many different doctors did I visit?",
        [
            _chunk("User: I visited my primary care physician for a checkup."),
            _chunk("User: I saw an ENT specialist about my sinus issues."),
            _chunk("User: I had a follow-up appointment with my dermatologist, Dr. Lee."),
            _chunk("Assistant: You should consult with your healthcare provider."),
        ],
    )

    assert result is not None
    assert result.operator == "count_health_events"
    assert result.answer.startswith("3")


def test_count_health_visits_prefers_doctor_types_over_names() -> None:
    result = ops.op_count_health_visits(
        "How many different doctors did I visit?",
        [
            _chunk("User: I had a follow-up appointment with my dermatologist, Dr. Lee."),
            _chunk("User: I later mentioned the follow-up with Dr. Lee for the biopsy."),
            _chunk("User: I visited my primary care physician."),
            _chunk("User: I saw an ENT specialist."),
        ],
    )

    assert result is not None
    assert result.answer.startswith("3")


def test_count_health_visits_counts_prescription_contact_but_not_future_schedule() -> None:
    result = ops.op_count_health_visits(
        "How many different doctors did I visit?",
        [
            _chunk("User: I was prescribed antibiotics by my primary care physician, Dr. Smith."),
            _chunk("User: I'll schedule a follow-up appointment with my primary care physician."),
            _chunk("User: I saw Dr. Patel, the ENT specialist, about sinusitis."),
        ],
    )

    assert result is not None
    assert result.answer.startswith("2")


def test_count_health_visits_counts_march_appointments() -> None:
    result = ops.op_count_health_visits(
        "How many doctor's appointments did I go to in March?",
        [
            _chunk("User: I had a follow-up appointment with Dr. Thompson on March 20th."),
            _chunk("User: I went to an ENT specialist appointment on March 7th."),
            _chunk("User: I am thinking about getting a colonoscopy in April."),
        ],
    )

    assert result is not None
    assert result.answer == "2"


def test_count_health_visits_rejects_medical_advice_without_visit() -> None:
    result = ops.op_count_health_visits(
        "How many different doctors did I visit?",
        [
            _chunk("Assistant: A dermatologist can help with suspicious moles."),
            _chunk("Assistant: Ask your doctor about colonoscopy risks."),
            _chunk("**Follow up with your ENT specialist**: Regular appointments can help."),
        ],
    )

    assert result is None


def test_current_reading_uses_latest_personal_current_book() -> None:
    result = ops.op_current_reading(
        "What book am I currently reading?",
        [
            _chunk('User: I am currently devouring "The Seven Husbands of Evelyn Hugo".'),
            _chunk("Assistant: You could read The Last House Guest next."),
        ],
    )

    assert result is not None
    assert result.operator == "current_state"
    assert result.answer == "The Seven Husbands of Evelyn Hugo"


def test_current_storage_location_uses_latest_storage_evidence() -> None:
    old = _chunk("User: I have been keeping my old sneakers under my bed for storage.")
    old.session_date = "2026/05/20"
    new = _chunk(
        "User: I need to organize my closet this weekend, and I'm looking forward to "
        "storing my old sneakers in a shoe rack in it."
    )
    new.session_date = "2026/05/22"

    result = ops.op_current_storage_location(
        "Where do I currently keep my old sneakers?",
        [old, new],
    )

    assert result is not None
    assert result.answer == "in shoe rack in my closet"


def test_airline_order_sorts_actual_flight_events_and_rejects_plans() -> None:
    chunks = [
        _chunk("User: I just got back from a red-eye flight on JetBlue from San Francisco to Boston."),
        _chunk("User: I am planning a trip and considering Spirit Airlines baggage policies."),
        _chunk("User: I just earned miles on my Delta SkyMiles card after taking a round-trip flight from Boston to Atlanta today."),
        _chunk("User: I had a 1-hour delay on my United Airlines flight from Boston to Chicago today."),
        _chunk("User: I had a terrible experience with American Airlines' entertainment system on my flight from New York to Los Angeles today."),
    ]
    for date, chunk in zip(
        ["2022/11/17", "2022/12/01", "2023/01/15", "2023/01/28", "2023/02/10"],
        chunks,
    ):
        chunk.session_date = date

    result = ops.op_airline_order(
        "What is the order of airlines I flew with from earliest to latest before today?",
        chunks,
    )

    assert result is not None
    assert result.answer == "JetBlue, Delta, United, American Airlines"


def test_count_named_attended_movie_festivals() -> None:
    result = ops.op_count_named_attended_events(
        "How many movie festivals that I attended?",
        [
            _chunk("User: I participated in the 48-hour film challenge at the Austin Film Festival."),
            _chunk("User: I volunteered at the Portland Film Festival and helped with coordination."),
            _chunk("User: I just got back from AFI Fest in LA, where I attended a screening."),
            _chunk("User: Can you recommend films that screen at festivals?"),
        ],
    )

    assert result is not None
    assert result.operator == "count_named_events"
    assert result.answer.startswith("3")


def test_count_named_attended_weddings_excludes_own_planning() -> None:
    result = ops.op_count_named_attended_events(
        "How many weddings have I attended in this year?",
        [
            _chunk("User: I'm planning my own wedding and need venue ideas."),
            _chunk("User: I just got back from Rachel and Mike's wedding at a vineyard."),
            _chunk("User: My friend Emily finally got to tie the knot with Sarah."),
            _chunk("User: At the wedding, the bride, Jen, looked stunning, and her husband, Tom, was happy."),
        ],
    )

    assert result is not None
    assert result.answer.startswith("3")


def test_current_numeric_status_uses_latest_followers_and_page() -> None:
    followers = ops.op_current_numeric_status(
        "How many followers do I have on Instagram now?",
            [
                _chunk("User: I've got 1250 followers on Instagram now."),
                _chunk("User: My current follower count is close to 1300 now."),
            ],
        )
    pages = ops.op_current_numeric_status(
        "How many pages of 'A Short History of Nearly Everything' have I read so far?",
        [
            _chunk('User: I have been reading "A Short History of Nearly Everything" and I am currently on page 200.'),
            _chunk('User: I picked up "A Short History of Nearly Everything" again and I am currently on page 220.'),
        ],
    )

    assert followers is not None
    assert followers.answer == "1300"
    assert pages is not None
    assert pages.answer == "220"


def test_current_numeric_status_prefers_user_star_correction() -> None:
    result = ops.op_current_numeric_status(
        "How many stars do I need to reach the gold level on my Starbucks Rewards app?",
        [
            _chunk("Assistant: To reach Gold level, you need 300 stars."),
            _chunk("User: Actually, I need 120 stars to reach the gold level, not 300."),
        ],
    )

    assert result is not None
    assert result.answer == "120"


def test_current_numeric_status_rejects_ranges_and_increase_questions() -> None:
    increase = ops.op_current_numeric_status(
        "What was the approximate increase in Instagram followers I experienced in two weeks?",
        [_chunk("User: I had around 350 followers. User: I reached about 600 followers two weeks later.")],
    )
    polluted = ops.op_current_numeric_status(
        "How many followers do I have on Instagram now?",
        [
            _chunk(
                "User: Can you suggest influencers? Assistant: Micro-influencers usually have "
                "1,000 - 10,000 followers."
            )
        ],
    )
    previous_best = ops.op_current_numeric_status(
        "What was my previous personal best time for the charity 5K run?",
        [_chunk("User: I improved my personal best from 27:45 to 25:50.")],
    )

    assert increase is None
    assert polluted is None
    assert previous_best is None
