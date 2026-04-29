MATCH (p:Person)
// EXISTS: The person's posts attract a diverse audience, confirmed by a "like"
// from someone using a different browser.
WHERE EXISTS {
  MATCH (p)<-[:HAS_CREATOR]-(:Post)<-[:LIKES]-(liker:Person)
  WHERE p.browserUsed <> liker.browserUsed
}
AND NOT EXISTS {
  MATCH (comment:Comment)-[:REPLY_OF]->(:Post)-[:HAS_CREATOR]->(p)
  WHERE comment.locationIP = p.locationIP
}
RETURN p.id;
