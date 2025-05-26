# Datasets

## census-1994.csv
Census data from USA gathered in 1994. Contains demographic information
about US citizens. These values are self-reported by the participating
citizens.

- age: Age of the person
- workclass: Type of employment of the person
- education: Education of the person
- marital_status: Civil status of the person
- occupation: The person's job
- relationship: The person's position in a family
- race: Note: The race category in US censuses is self-reported but with
  given possible values. These have changed over time.
- gender: This value is also restricted by the census form.
- capital_gain: How much the person made selling assets
- capital_loss: How much the person lost selling assets
- hours_per_week: How many hours the person works per week
- native_country: The native country of the person
- income_bracket: Whether the person earns more or less than 50.000 USD
  a year

## dolphins.csv
Data on dolphin behaviour in a narrow seaport.

- speed: Swimming speed of the dolphin (log10 transformation of km/h)
- rr: Reorientation rate - how much the dolphin changed
  direction.
- lin: Linearity - how much the movement of the dolphin deviated from a
  straight line with 0 representing no net movement and 1 representing moving in a straight line.
- distance: Unknown
- timeper: Time of day of the observation in a range from 0 to 1 where
  0-0.33 is morning, 0.34-0.66 is mid-day and 0.67-1 is the afternoon.
- cat: Category of boats or other vessels present near the dolphins (TT
  indicates the presence of both tour boats and trawlers).
- grpsize: Size of the group of dolphins.
- calf: Whether there were dolphin calves present.
- behav: Behaviour of the dolphins (FSB indicates foraging near a shrimp
  trawler).
- count: Unknown, doesn't seem to be the number of boats
- id: Individual observations (focal follows). Each observation is
  broken down into 10-minute segments.
