/**
  * Created by liushengchen on 4/28/17.
  */
import load.review
import org.scalatest.{FlatSpec, Matchers}


class reviewSpec extends  FlatSpec with Matchers{

  behavior of "review"

  it should "get the correct prediction" in {
    val prediction = review.FG
    prediction shouldBe 0.5 +- 0.5

  }
}
