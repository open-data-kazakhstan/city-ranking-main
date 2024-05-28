from datapackage import Package


package = Package()
package.infer(r'data/final_rating.csv')
package.commit()
package.save('datapackage.json')