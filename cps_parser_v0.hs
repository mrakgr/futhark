type Result = Either String String
type CPS r = (r -> String -> Result)
type Parser r = String -> CPS r -> CPS r -> Result

win_fin r _ = Right $ "Correct: " ++ show r
lose_fin _ _ = Left $ "Failed"

parse_a :: Parser Char
parse_a (x:xs) succ fail =
  if x == 'a' then succ x xs else fail x xs

parse_many :: Parser Char -> Parser String
parse_many parser = \text win_cps lose_cps ->
  loop [] text win_cps lose_cps where
    loop :: [Char] -> Parser String
    loop l = \text win_cps lose_cps ->
      let win r new_text = loop (r:l) new_text win_cps lose_cps
          lose _ = win_cps l in
      parser text win lose

main = print $ (parse_many parse_a) "aaabc" win_fin lose_fin
