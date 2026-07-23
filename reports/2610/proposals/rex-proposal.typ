// =============================================================================
// PREAMBLE: Global Document Rules & Variables
// =============================================================================

// Default Global Spacing
#let global-spacing = 11pt
#let table-inset = 6pt
#let line-height = 0.5pt

// Page Setup
#set page(
  paper: "a4",
  margin: (x: 2.5cm, top: 2.5cm, bottom: 2.5cm),
  header: text(fill: luma(100))[
    #grid(
      columns: (1fr, auto),
      [Office of the Provost \ National University of Singapore],
      align(right)[School of Computing \ Updated 11 Jun 2024],
    )
  ],
  footer: align(center)[
    // #text(fill: luma(100))[Page #counter(page).display()]
  ],
)

// Text Setup (Arial, 10pt uniform throughout)
#set text(
  font: "Arial",
  size: 10pt,
  hyphenate: false,
)

// Paragraph & Block Rules
#set par(spacing: global-spacing, leading: 1em, justify: true)
#set block(spacing: global-spacing)

#show heading.where(level: 1): set text(size: 10pt)

// Form Underline Field Helper
#let field(label, content) = {
  block(
    width: 100%,
    stroke: (bottom: line-height + black),
    inset: (bottom: 0.30em),
    [ _#label:_ #content ],
  )
}

// =============================================================================
// DOCUMENT BODY
// =============================================================================

#align(center)[
  *REx Fellows’ Grant Proposal Application Form*
]

#line(length: 100%, stroke: line-height + black)

#text(fill: luma(50%))[_
*Note to applicants:*
- All fields must be completed. Incomplete forms may not be accepted.
- Applicants are to submit the grant proposal form by the deadline stipulated in PDF format to Microsoft form at #link("https://forms.office.com")[REx Application].
- All text should be in single spacing, normal margin, Arial, 10 pt.
- Prior to submission, applicants should obtain the endorsement of their supervisor. Applicants should also copy their supervisor when submitting the form.
- Applicants and supervisors may refer to the REx Fellows’ Grant Proposal Application Guidelines for details on eligibility and terms and conditions.
- For enquiries related to REx, please contact the Undergraduate Research Coordinating Team:
  - NUS REx at #link("mailto:pvobox66@nus.edu.sg")[pvobox66\@nus.edu.sg]
_]

#line(length: 100%, stroke: line-height + black)

#v(1em)
= Applicant’s details

#field("Name of Applicant and Student Identification Number", [John Doe / A0123456X])

#grid(
  columns: (1fr, 1fr),
  gutter: 15pt,
  field("Student NUS email", [e0123456\@u.nus.edu]), field("Major/Year of Study", [Computer Science / Year 3]),
)

#v(1em)
= Details of UROP project

#grid(
  columns: (1fr, 1fr),
  gutter: 15pt,
  field("Course Code", [CP3209]), field("Units", [8 Units]),
)

#field("UROP project code and title", [])

#field("Department of UROP project", [Department of Computer Science])

#grid(
  columns: (1fr, 1fr),
  gutter: 15pt,
  field("Start of UROP project", [AY26/27, Sem 1]), field("Grant Application Semester", [AY26/27, Sem 1]),
)

#v(1em)
= Name, email and affiliated department of UROP supervisor(s)

#field("Main Supervisor", [])
#field("Co-Supervisor", [])
#field("Date of submission", [])

#pagebreak()

// --- Section 2: Research Grant Proposal ---

#align(center)[
  #title[Research Grant Proposal]
]

In 1 page or more, include the following sections in the grant proposal:

*(a) UROP+REx project title:* 

*(b) Proposed duration of project (maximum duration of REx is 12 months)*
- *(i) Start Date:* 01/08/2026
- *(ii) End Date:* 31/07/2027
- *(iii) Duration of UROP+REx project (months):* 12 Months
- *(iv) Abstract of UROP+REx project (no more than 100 words):*
  Insert project abstract here...

*(b) More details about the project, including the proposed deliverables*
Refer to the REx Fellows’ Grant Proposal Application Guidelines on possible deliverables.

*(c) Proposed budget breakdown and justification*

*Note:*
1. Applicant should discuss with REx supervisor(s) on the overall proposed budget. Up to \$2,500 research grant will be provided to support the cost of research work and deliverables for UROP+REx project. Applicants and potential supervisors may refer to REx Fellows’ Grant Proposal Application Guidelines for details pertaining to the terms and conditions of the grant.
2. Include the following table for this section (e):

#table(
  columns: (auto, 1fr, 2fr),
  inset: table-inset,
  stroke: 0.5pt + black,
  [*Item*], [*Proposed budget (SGD)*], [*Justification*],
  [1)], [\$1,000], [High-Performance GPU Cloud Compute Services],
  [2)], [\$500], [Conference Registration Fees and Poster Printing],
  [*Total:*], [\$1,500], [],
)

*(d) References and appendices (if any)*

I confirm that my supervisor(s) have agreed for the UROP course to be REx-upgraded: *Yes / No*

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 15pt,
  [
    #line(length: 100%, stroke: line-height + black)
    *Signature of applicant*
  ],
  [
    #line(length: 100%, stroke: line-height + black)
    *Signature of supervisor(s)*
  ],
  [
    #line(length: 100%, stroke: line-height + black)
    *Date*
  ],
)

_You may delete this line and insert the Research Grant Proposal from this page onwards, which should be 1 page or more._

#pagebreak()

// --- Section 3: Guidelines ---

#align(center)[
  *REx Fellows’ Grant Proposal Application Guidelines (SoC)*
]

*1. Introduction*

1.1 The Undergraduate Research Opportunities Programme (UROP) launched the Research Experience (REx) course upgrade so as to fulfil the following objectives:
- (a) To provide more avenues for undergraduates to experience deeper immersion and training in research.
- (b) Streamlining and optimizing the administrative and approval processes in order to facilitate participation and ease any possible difficulties.
- (c) To attract more faculty members and full-time researchers to supervise undergraduate research projects and foster a culture of undergraduate research; and
- (d) To provide better support to all participants and relevant stakeholders in the undergraduate research ecosystem.

1.2 REx is an upgrade of the existing UROP course. Students who are interested to enrol in REx will submit a grant proposal to be eligible for a research grant of up to \$2,500 which can be used to reimburse costs of the associated UROP project.

1.3 The research grant will be provided on a requirement basis to the UROP+REx project. Each grant proposal submitted should account for the expenditures involved in the utilisation of the grant and an abstract pertaining to their expected research deliverables in their existing UROP projects when enrolled in REx. These deliverables may include (but not limited to):
- (a) Presentation of working papers at workshops or special REx seminars open to all NUS staff and students.
- (b) Presentation of working posters at poster sessions or internal undergraduate research symposium or conference.

*2. Eligibility criteria*

2.1 All eligibility criteria for UROP applies to REx. Students must be registered under SoC, have at least 60 Units at the point of applying (including units enrolled during project selection semester).

2.2 Students can enrol in REx only once during their undergraduate studies. REx Fellows who have received the REx Fellowship Certificate are not eligible to apply for REx again.

*3. Application Procedure and Approval of Project Grant*

3.1 The grant call for REx grant proposals will be held twice a year. The grant call and closing dates will be communicated to applicants by the UG Research Coordinating team at the Office of the Provost (PVO).

3.2 Each grant proposal will be evaluated by the UG Research Coordinating team according to the following criteria:
- (a) Clear articulation of motivations for seeking to enrol in REx.
- (b) Clear articulation of research gaps and proposed methods to address these gaps in the associated UROP project.
- (c) Justifiable proposed budget breakdown of REx research grant.

3.3 Applicants are to fill out all sections of the REx Fellows’ Grant Proposal Application Form containing applicant’s details, description of the project and a tabulated proposed budget breakdown. The completed form should be submitted to the department course coordinator for UROP by email as a PDF file named as “Studentname_Department_SoC_MonthYear.pdf”, where “Department” refers to the department hosting the UROP project.

3.4 Applicants will be informed of their application outcome by email. Upon approval of the grant proposal, applicants will be named as a “REx Fellow” and will be allocated REX1000 on their CANVAS page.

*4. Terms and Conditions*

4.1 Supervisors have the responsibility to ensure that the project carried out by REx Fellows are in compliance with NUS’ code of ethics, research integrity and responsible research conduct.

4.2 Wherever applicable, supervisors should ascertain that necessary approvals are in place before carrying out the research activities.

4.3 The REx Fellows’ research grant shall be used solely for the purposes of the associated UROP project. REx Fellows from SoC have until their last semester before graduation to utilise the grant monies.

4.4 The following items are allowed to be claimed from the REx Fellows’ research grant:
- (a) Consumables, lab supplies, use of research facilities and equipment directly required for the associated UROP project.
- (b) License fees of software to be used for the associated UROP project.
- (c) Registration fees for conferences, seminars or workshops.
- (d) Costs of preparing project deliverables such as costs of printing posters, travel and transportation costs directly associated to carrying out the associated UROP project.
- (e) Books or journal articles which are directly related to the associated UROP project. Subscription to journals which are NOT available in the university’s libraries is allowed.
- (f) Fees associated with publications of manuscript in journals.

4.5 The ownership of items purchased using the REx Fellows’ research grant will be subject to agreements between REx Fellows and their supervisors. Items such as books, journal articles or posters can be kept by REx Fellows with the approval of their supervisors. Items to be used in a lab setting should be kept by the research lab.

4.6 The following items are not allowed to be claimed from the REx Fellows’ research grant:
- (a) Entertainment and refreshment.
- (b) Personal expenses not related to associated UROP project.
- (c) Equipment or devices not related to associated UROP project.
