"""Plain-language meaning of every word CivetWatch shows: used by the "Read me"
tab of sheet.xlsx and the "What the words mean" box of the dashboard."""

GLOSSARY = [
    ("Inactive", "The civet is resting: lying down, stretched out, curled up or sleeping in a box. "
                 "It only switches between Inactive and Active after 1 minute or more, so a short stretch "
                 "or a few steps do not count as a change."),
    ("Active", "The civet is awake and doing something: walking, eating, drinking, sniffing, exploring."),
    ("Not visible", "No camera can see the civet (for example it is sleeping in a spot outside every camera's view). "
                    "This time is neither Active nor Inactive - we do not guess."),
    ("Not seen", "Same as Not visible, in the 'time per room' chart."),
    ("Active 84 % of the time it was seen", "Share of the visible time that was Active. Not-visible time is left out."),
    ("Feeding / Drinking (visits)", "The civet stays at the food dish / water basin with its head at it. "
                                    "Looking up and going back within 1 minute is the same visit; "
                                    "walking past the bowl does not count."),
    ("Grooming", "Licking, nibbling or scratching its own fur."),
    ("Territorial marking (scent marking)", "Rubbing or pressing its rear end - where the scent gland is - "
                                            "on a log, wall, door frame or fence."),
    ("Urine", "Urinating."),
    ("Feces", "Defecating."),
    ("Short stop", "The civet walked, stood still for 4-60 seconds, then walked on. Grooming, marking and toileting "
                   "happen in these stops, so every one is shown to a person to check (dashboard > Quick check)."),
    ("Passing between rooms", "A short stop in a doorway while the civet moves to the next room - not checked."),
    ("Please check / hollow dot", "The app is not sure. Look at the clip or pictures and press Correct, "
                                  "Not this behavior, or pick what it really was."),
    ("Counted / filled dot", "Included in the observation sheet."),
    ("Confidence", "How sure the civet detector is that the box really is the civet (0 to 1)."),
    ("night IR / day colour", "Night: the camera films in black and white infrared. Day: normal colour."),
    ("Poor image", "The picture was blurred, too dark or frozen - not used for counting."),
    ("Spot check", "Random moments of the day that you answer without seeing the app's answer; "
                   "it gives the real accuracy of the app."),
]
